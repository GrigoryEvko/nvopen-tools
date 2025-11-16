// Function: sub_2B1CDB0
// Address: 0x2b1cdb0
//
__int64 __fastcall sub_2B1CDB0(
        unsigned int *src,
        unsigned int *a2,
        char *a3,
        __int64 a4,
        __int64 a5,
        __m128i a6,
        __int64 a7,
        __int128 a8,
        __int64 a9)
{
  __int64 v11; // rax
  unsigned int *v12; // r15
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // [rsp+8h] [rbp-38h]

  v11 = (a2 - src + 1) / 2;
  v12 = &src[v11];
  v17 = 4 * v11;
  if ( v11 <= a4 )
  {
    sub_2B1C380(src, &src[v11], a3, a4, a5, 4 * v11, a6, a8);
    sub_2B1C380(v12, a2, a3, v14, v15, v16, a6, a8);
  }
  else
  {
    sub_2B1CDB0(src, a8, SDWORD2(a8), a9);
    sub_2B1CDB0(v12, a8, SDWORD2(a8), a9);
  }
  return sub_2B1C790(src, v12, (__int64)a2, v17 >> 2, a2 - v12, (unsigned int *)a3, a4, a8, a9);
}
