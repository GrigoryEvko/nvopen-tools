// Function: sub_3249E90
// Address: 0x3249e90
//
__int64 __fastcall sub_3249E90(__int64 *a1, __int64 a2, char a3, __int64 a4)
{
  int v5; // [rsp+0h] [rbp-4h]

  LOWORD(v5) = a3 == 0 ? 13 : 15;
  BYTE2(v5) = 1;
  return sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 28, v5, a4);
}
