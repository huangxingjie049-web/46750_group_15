
from data_interface import load_q1_data, convert_member
from follower import solve_member
from kkt_validation import validate_kkt


def main():
    # Load Q1 community data
    data = load_q1_data()
    prices = list(data.prices)

    # Select members with and without battery
    members = list(data.members)

    for source in members:
        member = convert_member(source)

        print("\n" + "=" * 65)
        print("Member:", source.id)
        print("Has battery:", member["battery"] is not None)

        # Solve follower and retain the Gurobi model.
        result = solve_member(
            member,
            prices,
            output=False,
            return_model=True
        )

        print(
            "Follower objective:",
            round(result["objective"], 6),
            "DKK"
        )

        validate_kkt(result["_model"])

    print("\nBoth follower KKT validations completed.")


if __name__ == "__main__":
    main()
