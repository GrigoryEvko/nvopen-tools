// Function: sub_1026850
// Address: 0x1026850
//
__int64 __fastcall sub_1026850(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  char v9; // r15
  __int64 v11; // [rsp+0h] [rbp-60h]
  char v12; // [rsp+Fh] [rbp-51h]
  __int64 v15[7]; // [rsp+28h] [rbp-38h] BYREF

  v11 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 72LL);
  v15[0] = sub_B2D7E0(v11, "no-nans-fp-math", 0xFu);
  v12 = sub_A72A30(v15);
  v15[0] = sub_B2D7E0(v11, "no-signed-zeros-fp-math", 0x17u);
  v9 = (2 * v12) | (8 * sub_A72A30(v15));
  if ( (unsigned __int8)sub_1024C90(a1, 1, a2, v9, a3, a4, a5, a6, a7)
    || (unsigned __int8)sub_1024C90(a1, 2, a2, v9, a3, a4, a5, a6, a7)
    || (unsigned __int8)sub_1024C90(a1, 3, a2, v9, a3, a4, a5, a6, a7)
    || (unsigned __int8)sub_1024C90(a1, 4, a2, v9, a3, a4, a5, a6, a7)
    || (unsigned __int8)sub_1024C90(a1, 5, a2, v9, a3, a4, a5, a6, a7)
    || (unsigned __int8)sub_1024C90(a1, 7, a2, v9, a3, a4, a5, a6, a7)
    || (unsigned __int8)sub_1024C90(a1, 6, a2, v9, a3, a4, a5, a6, a7)
    || (unsigned __int8)sub_1024C90(a1, 9, a2, v9, a3, a4, a5, a6, a7)
    || (unsigned __int8)sub_1024C90(a1, 8, a2, v9, a3, a4, a5, a6, a7)
    || (unsigned __int8)sub_1024C90(a1, 17, a2, v9, a3, a4, a5, a6, a7)
    || (unsigned __int8)sub_1024C90(a1, 19, a2, v9, a3, a4, a5, a6, a7)
    || (unsigned __int8)sub_1024C90(a1, 11, a2, v9, a3, a4, a5, a6, a7)
    || (unsigned __int8)sub_1024C90(a1, 10, a2, v9, a3, a4, a5, a6, a7)
    || (unsigned __int8)sub_1024C90(a1, 13, a2, v9, a3, a4, a5, a6, a7)
    || (unsigned __int8)sub_1024C90(a1, 12, a2, v9, a3, a4, a5, a6, a7)
    || (unsigned __int8)sub_1024C90(a1, 18, a2, v9, a3, a4, a5, a6, a7)
    || (unsigned __int8)sub_1024C90(a1, 16, a2, v9, a3, a4, a5, a6, a7)
    || (unsigned __int8)sub_1024C90(a1, 15, a2, v9, a3, a4, a5, a6, a7) )
  {
    return 1;
  }
  else
  {
    return sub_1024C90(a1, 14, a2, v9, a3, a4, a5, a6, a7);
  }
}
