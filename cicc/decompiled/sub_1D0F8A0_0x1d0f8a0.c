// Function: sub_1D0F8A0
// Address: 0x1d0f8a0
//
__int64 __fastcall sub_1D0F8A0(int *a1, unsigned int *a2, int *a3, __int64 a4)
{
  __int64 v6; // r9
  int *v7; // r15
  __int64 v9; // [rsp-10h] [rbp-50h]
  __int64 v10; // [rsp+8h] [rbp-38h]

  v6 = 4 * (((((char *)a2 - (char *)a1) >> 4) + 1) / 2);
  v10 = v6 * 4;
  v7 = &a1[v6];
  if ( ((((char *)a2 - (char *)a1) >> 4) + 1) / 2 <= a4 )
  {
    sub_1D0C130(a1, (unsigned int *)&a1[v6], a3);
    sub_1D0C130(v7, a2, a3);
  }
  else
  {
    sub_1D0F8A0(a1, &a1[v6], a3);
    sub_1D0F8A0(v7, a2, a3);
  }
  sub_1D0F4E0((char *)a1, (__int64)v7, (__int64)a2, v10 >> 4, ((char *)a2 - (char *)v7) >> 4, a3, a4);
  return v9;
}
