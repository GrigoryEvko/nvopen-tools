// Function: sub_17E66F0
// Address: 0x17e66f0
//
__int64 __fastcall sub_17E66F0(char *a1, char *a2, __int64 *a3, __int64 a4)
{
  __int64 v6; // rax
  char *v7; // r15
  __int64 v9; // [rsp-10h] [rbp-50h]
  __int64 v10; // [rsp+8h] [rbp-38h]

  v6 = (((a2 - a1) >> 3) + 1) / 2;
  v10 = 8 * v6;
  v7 = &a1[8 * v6];
  if ( v6 <= a4 )
  {
    sub_17E36B0(a1, &a1[8 * v6], a3);
    sub_17E36B0(v7, a2, a3);
  }
  else
  {
    sub_17E66F0(a1, &a1[8 * v6], a3);
    sub_17E66F0(v7, a2, a3);
  }
  sub_17E5E90((__int64)a1, (__int64)v7, (__int64)a2, v10 >> 3, (a2 - v7) >> 3, (char *)a3, a4);
  return v9;
}
