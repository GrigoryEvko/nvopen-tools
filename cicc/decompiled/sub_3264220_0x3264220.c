// Function: sub_3264220
// Address: 0x3264220
//
__int64 __fastcall sub_3264220(unsigned int *src, unsigned int *a2, char *a3, __int64 a4)
{
  __int64 v6; // r9
  unsigned int *v7; // r15
  __int64 v9; // [rsp-10h] [rbp-50h]
  __int64 v10; // [rsp+8h] [rbp-38h]

  v6 = 4 * (((((char *)a2 - (char *)src) >> 4) + 1) / 2);
  v10 = v6 * 4;
  v7 = &src[v6];
  if ( ((((char *)a2 - (char *)src) >> 4) + 1) / 2 <= a4 )
  {
    sub_3261780(src, &src[v6], a3);
    sub_3261780(v7, a2, a3);
  }
  else
  {
    sub_3264220(src);
    sub_3264220(v7);
  }
  sub_3263A70(src, v7, (__int64)a2, v10 >> 4, ((char *)a2 - (char *)v7) >> 4, (unsigned int *)a3, a4);
  return v9;
}
