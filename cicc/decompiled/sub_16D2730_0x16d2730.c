// Function: sub_16D2730
// Address: 0x16d2730
//
unsigned __int64 __fastcall sub_16D2730(const __m128i *a1, __int64 a2, char *a3, size_t a4, int a5, __int64 a6)
{
  unsigned __int64 result; // rax
  int v9; // ebx
  __int64 v10; // rdx
  __int64 v11; // r10
  __int64 v12; // rdx
  __int64 *v13; // rdx
  __int64 v14; // rax
  __int64 v15; // [rsp+8h] [rbp-68h]
  __int64 v16; // [rsp+10h] [rbp-60h]
  unsigned __int64 v17; // [rsp+18h] [rbp-58h]
  char v18; // [rsp+28h] [rbp-48h]
  char v19; // [rsp+2Fh] [rbp-41h]
  __m128i v20; // [rsp+30h] [rbp-40h] BYREF

  result = a2 + 16;
  v18 = a6;
  v19 = a6;
  v20 = _mm_loadu_si128(a1);
  if ( a5 )
  {
    v9 = a5;
    do
    {
      result = sub_16D20C0(v20.m128i_i64, a3, a4, 0);
      if ( result == -1 )
        break;
      if ( result || v19 )
      {
        v10 = v20.m128i_i64[1];
        a6 = 0;
        if ( result )
        {
          if ( result <= v20.m128i_i64[1] )
            v10 = result;
          a6 = v10;
        }
        v11 = v20.m128i_i64[0];
        v12 = *(unsigned int *)(a2 + 8);
        if ( (unsigned int)v12 >= *(_DWORD *)(a2 + 12) )
        {
          v15 = v20.m128i_i64[0];
          v16 = a6;
          v17 = result;
          sub_16CD150(a2, (const void *)(a2 + 16), 0, 16, a5, (unsigned __int8)a6);
          v12 = *(unsigned int *)(a2 + 8);
          v11 = v15;
          a6 = v16;
          result = v17;
        }
        v13 = (__int64 *)(*(_QWORD *)a2 + 16 * v12);
        *v13 = v11;
        v13[1] = a6;
        ++*(_DWORD *)(a2 + 8);
      }
      result += a4;
      if ( result > v20.m128i_i64[1] )
        result = v20.m128i_u64[1];
      v20.m128i_i64[0] += result;
      v20.m128i_i64[1] -= result;
      --v9;
    }
    while ( v9 );
  }
  if ( v18 || v20.m128i_i64[1] )
  {
    v14 = *(unsigned int *)(a2 + 8);
    if ( (unsigned int)v14 >= *(_DWORD *)(a2 + 12) )
    {
      sub_16CD150(a2, (const void *)(a2 + 16), 0, 16, a5, (unsigned __int8)a6);
      v14 = *(unsigned int *)(a2 + 8);
    }
    result = *(_QWORD *)a2 + 16 * v14;
    *(__m128i *)result = _mm_load_si128(&v20);
    ++*(_DWORD *)(a2 + 8);
  }
  return result;
}
