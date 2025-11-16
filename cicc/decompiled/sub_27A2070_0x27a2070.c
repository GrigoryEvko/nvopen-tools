// Function: sub_27A2070
// Address: 0x27a2070
//
__int64 __fastcall sub_27A2070(int *a1, int *a2, char *a3, __int64 a4)
{
  __int64 v6; // r9
  int *v7; // r15
  __int64 v9; // [rsp-10h] [rbp-50h]
  __int64 v10; // [rsp+8h] [rbp-38h]

  v6 = 8 * (((((char *)a2 - (char *)a1) >> 5) + 1) / 2);
  v10 = v6 * 4;
  v7 = &a1[v6];
  if ( ((((char *)a2 - (char *)a1) >> 5) + 1) / 2 <= a4 )
  {
    sub_27A1690(a1, &a1[v6], a3);
    sub_27A1690(v7, a2, a3);
  }
  else
  {
    sub_27A2070(a1, &a1[v6], a3);
    sub_27A2070(v7, a2, a3);
  }
  sub_27A1980((char *)a1, (__int64)v7, (__int64)a2, v10 >> 5, ((char *)a2 - (char *)v7) >> 5, a3, a4);
  return v9;
}
