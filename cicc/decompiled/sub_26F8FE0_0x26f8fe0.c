// Function: sub_26F8FE0
// Address: 0x26f8fe0
//
__int64 __fastcall sub_26F8FE0(
        __int64 *a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 *a4,
        __int64 a5,
        unsigned int a6,
        __m128i a7,
        unsigned int *a8)
{
  __int64 v12; // rdi
  unsigned int *v13; // r15
  __int64 result; // rax
  __int64 **v15; // r15
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  __m128i v18; // [rsp+0h] [rbp-50h] BYREF
  unsigned __int64 v19; // [rsp+10h] [rbp-40h]
  __int64 v20; // [rsp+18h] [rbp-38h]

  v12 = *a1;
  v20 = a2;
  v19 = a3;
  v13 = a8;
  v18 = _mm_loadu_si128(&a7);
  result = sub_26F8EA0(v12);
  if ( (_BYTE)result )
  {
    v15 = (__int64 **)a1[8];
    v16 = sub_ACD640(a1[9], a6, 0);
    v17 = sub_AD4C70(v16, v15, 0);
    a7 = _mm_load_si128(&v18);
    return sub_26F8F40(a1, v20, v19, a4, a5, v17, (unsigned __int8 *)a7.m128i_i64[0], a7.m128i_u64[1]);
  }
  else
  {
    *v13 = a6;
  }
  return result;
}
