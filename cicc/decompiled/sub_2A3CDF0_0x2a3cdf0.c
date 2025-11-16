// Function: sub_2A3CDF0
// Address: 0x2a3cdf0
//
unsigned __int64 __fastcall sub_2A3CDF0(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  const void *v4; // r13
  unsigned __int64 v5; // r15
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r15
  unsigned __int64 result; // rax
  __int64 v10; // rax
  __m128i v11; // xmm0
  __int64 v12; // r9
  __m128i v13; // [rsp+0h] [rbp-80h] BYREF
  __int64 v14; // [rsp+18h] [rbp-68h]
  __int64 v15; // [rsp+20h] [rbp-60h] BYREF
  unsigned __int64 i; // [rsp+28h] [rbp-58h]
  __m128i v17; // [rsp+30h] [rbp-50h] BYREF

  v4 = (const void *)(a3 + 16);
  v15 = a1;
  for ( i = a2; ; i = v7 )
  {
    v17.m128i_i8[0] = 44;
    result = sub_C931B0(&v15, &v17, 1u, 0);
    if ( result != -1 )
      break;
    result = i;
    v6 = v15;
    v7 = 0;
    v8 = 0;
    if ( !i )
      return result;
LABEL_7:
    v17.m128i_i64[1] = result;
    v10 = *(unsigned int *)(a3 + 8);
    v17.m128i_i64[0] = v6;
    v11 = _mm_load_si128(&v17);
    if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
    {
      v14 = v7;
      v13 = v11;
      sub_C8D5F0(a3, v4, v10 + 1, 0x10u, v7, v12);
      v10 = *(unsigned int *)(a3 + 8);
      v11 = _mm_load_si128(&v13);
      v7 = v14;
    }
    *(__m128i *)(*(_QWORD *)a3 + 16 * v10) = v11;
    ++*(_DWORD *)(a3 + 8);
    v15 = v8;
  }
  v5 = result + 1;
  v6 = v15;
  if ( result + 1 > i )
  {
    v5 = i;
    v7 = 0;
  }
  else
  {
    v7 = i - v5;
  }
  v8 = v15 + v5;
  if ( result > i )
    result = i;
  if ( result )
    goto LABEL_7;
  return result;
}
