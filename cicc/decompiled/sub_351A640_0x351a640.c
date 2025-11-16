// Function: sub_351A640
// Address: 0x351a640
//
__int64 __fastcall sub_351A640(int *a1, unsigned int *a2, int *a3, __int64 a4)
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
    sub_3510BD0(a1, (unsigned int *)&a1[v6], a3);
    sub_3510BD0(v7, a2, a3);
  }
  else
  {
    sub_351A640(a1, &a1[v6], a3);
    sub_351A640(v7, a2, a3);
  }
  sub_351A280((char *)a1, (char *)v7, (__int64)a2, v10 >> 4, ((char *)a2 - (char *)v7) >> 4, a3, a4);
  return v9;
}
