// Function: sub_C937F0
// Address: 0xc937f0
//
_OWORD *__fastcall sub_C937F0(const __m128i *a1, __int64 a2, _WORD *a3, size_t a4, __int64 a5, __int64 a6)
{
  _OWORD *result; // rax
  int v10; // ebx
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // r10
  __int64 *v15; // rdx
  __int64 v16; // rax
  __m128i v17; // xmm0
  __int64 v18; // [rsp+8h] [rbp-78h]
  __int64 v19; // [rsp+10h] [rbp-70h]
  _OWORD *v20; // [rsp+18h] [rbp-68h]
  const void *v21; // [rsp+20h] [rbp-60h]
  char v22; // [rsp+2Ch] [rbp-54h]
  __m128i v23; // [rsp+30h] [rbp-50h] BYREF
  __m128i v24; // [rsp+40h] [rbp-40h] BYREF

  result = (_OWORD *)(a2 + 16);
  v22 = a6;
  v23.m128i_i8[0] = a6;
  v21 = (const void *)(a2 + 16);
  v24 = _mm_loadu_si128(a1);
  if ( (_DWORD)a5 )
  {
    v10 = a5;
    do
    {
      result = (_OWORD *)sub_C931B0(v24.m128i_i64, a3, a4, 0);
      if ( result == (_OWORD *)-1LL )
        break;
      if ( result || v23.m128i_i8[0] )
      {
        v13 = *(unsigned int *)(a2 + 8);
        a6 = (__int64)result;
        v14 = v24.m128i_i64[0];
        if ( v24.m128i_i64[1] <= (unsigned __int64)result )
          a6 = v24.m128i_i64[1];
        if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        {
          v18 = v24.m128i_i64[0];
          v19 = a6;
          v20 = result;
          sub_C8D5F0(a2, v21, v13 + 1, 0x10u, a5, a6);
          v13 = *(unsigned int *)(a2 + 8);
          v14 = v18;
          a6 = v19;
          result = v20;
        }
        v15 = (__int64 *)(*(_QWORD *)a2 + 16 * v13);
        *v15 = v14;
        v15[1] = a6;
        ++*(_DWORD *)(a2 + 8);
      }
      v11 = v24.m128i_i64[1];
      result = (_OWORD *)((char *)result + a4);
      v12 = 0;
      if ( (unsigned __int64)result <= v24.m128i_i64[1] )
      {
        v11 = (__int64)result;
        v12 = v24.m128i_i64[1] - (_QWORD)result;
      }
      v24.m128i_i64[1] = v12;
      v24.m128i_i64[0] += v11;
      --v10;
    }
    while ( v10 );
  }
  if ( v22 || v24.m128i_i64[1] )
  {
    v16 = *(unsigned int *)(a2 + 8);
    v17 = _mm_load_si128(&v24);
    if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      v23 = v17;
      sub_C8D5F0(a2, (const void *)(a2 + 16), v16 + 1, 0x10u, a5, a6);
      v16 = *(unsigned int *)(a2 + 8);
      v17 = _mm_load_si128(&v23);
    }
    result = (_OWORD *)(*(_QWORD *)a2 + 16 * v16);
    *result = v17;
    ++*(_DWORD *)(a2 + 8);
  }
  return result;
}
