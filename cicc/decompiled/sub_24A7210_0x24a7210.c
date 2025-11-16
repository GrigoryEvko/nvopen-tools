// Function: sub_24A7210
// Address: 0x24a7210
//
__int64 __fastcall sub_24A7210(unsigned __int64 *a1, unsigned __int64 *a2, unsigned __int64 *a3, __int64 a4)
{
  __int64 v6; // rax
  unsigned __int64 *v7; // r15
  __int64 v9; // [rsp-10h] [rbp-50h]
  __int64 v10; // [rsp+8h] [rbp-38h]

  v6 = (a2 - a1 + 1) / 2;
  v10 = 8 * v6;
  v7 = &a1[v6];
  if ( v6 <= a4 )
  {
    sub_24A3A30(a1, &a1[v6], a3);
    sub_24A3A30(v7, a2, a3);
  }
  else
  {
    sub_24A7210(a1, &a1[v6], a3);
    sub_24A7210(v7, a2, a3);
  }
  sub_24A69B0((__int64)a1, (__int64)v7, (__int64)a2, v10 >> 3, a2 - v7, a3, a4);
  return v9;
}
