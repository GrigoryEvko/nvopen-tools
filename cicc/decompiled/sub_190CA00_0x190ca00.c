// Function: sub_190CA00
// Address: 0x190ca00
//
__int64 __fastcall sub_190CA00(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 result; // rax
  __int64 v9; // r14
  __int64 v10; // r12
  int v11; // edx
  __int64 v12; // rsi
  unsigned int v13; // eax
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // eax
  __int64 *v18; // rcx
  __int64 v19; // rbx
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v25; // [rsp+10h] [rbp-70h]
  __int64 v27; // [rsp+20h] [rbp-60h] BYREF
  int v28; // [rsp+28h] [rbp-58h]
  __m128i v29; // [rsp+30h] [rbp-50h] BYREF
  __int64 v30; // [rsp+40h] [rbp-40h]

  result = *(unsigned int *)(a3 + 8);
  if ( (_DWORD)result )
  {
    v9 = a5;
    v10 = 0;
    v25 = 24 * result;
    while ( 1 )
    {
      v17 = *(_DWORD *)(a1 + 72);
      v18 = (__int64 *)(v10 + *(_QWORD *)a3);
      v19 = *v18;
      if ( v17 )
      {
        v11 = v17 - 1;
        v12 = *(_QWORD *)(a1 + 56);
        LODWORD(a5) = 1;
        v13 = (v17 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
        v14 = *(_QWORD *)(v12 + 8LL * v13);
        if ( v19 == v14 )
        {
LABEL_4:
          v29.m128i_i64[0] = *v18;
          v15 = *(unsigned int *)(a4 + 8);
          v29.m128i_i64[1] = 6;
          LODWORD(v30) = 0;
          if ( (unsigned int)v15 >= *(_DWORD *)(a4 + 12) )
          {
            sub_16CD150(a4, (const void *)(a4 + 16), 0, 24, a5, a6);
            v15 = *(unsigned int *)(a4 + 8);
          }
          result = *(_QWORD *)a4 + 24 * v15;
          v16 = v30;
          *(__m128i *)result = _mm_loadu_si128(&v29);
          *(_QWORD *)(result + 16) = v16;
          ++*(_DWORD *)(a4 + 8);
          goto LABEL_7;
        }
        while ( v14 != -8 )
        {
          a6 = a5 + 1;
          v13 = v11 & (a5 + v13);
          v14 = *(_QWORD *)(v12 + 8LL * v13);
          if ( v19 == v14 )
            goto LABEL_4;
          LODWORD(a5) = a5 + 1;
        }
      }
      v20 = v18[1];
      if ( (unsigned int)(v20 & 7) - 1 <= 1
        && (v21 = v18[2], v27 = 0, (unsigned __int8)sub_190C3B0(a1, a2, v20, v21, (__int64)&v27, 0)) )
      {
        v29.m128i_i64[0] = v19;
        v29.m128i_i64[1] = v27;
        LODWORD(v30) = v28;
        v22 = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)v22 >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, (const void *)(a4 + 16), 0, 24, a5, a6);
          v22 = *(unsigned int *)(a4 + 8);
        }
        v10 += 24;
        result = *(_QWORD *)a4 + 24 * v22;
        v23 = v30;
        *(__m128i *)result = _mm_loadu_si128(&v29);
        *(_QWORD *)(result + 16) = v23;
        ++*(_DWORD *)(a4 + 8);
        if ( v25 == v10 )
          return result;
      }
      else
      {
        result = *(unsigned int *)(v9 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(v9 + 12) )
        {
          sub_16CD150(v9, (const void *)(v9 + 16), 0, 8, a5, a6);
          result = *(unsigned int *)(v9 + 8);
        }
        *(_QWORD *)(*(_QWORD *)v9 + 8 * result) = v19;
        ++*(_DWORD *)(v9 + 8);
LABEL_7:
        v10 += 24;
        if ( v25 == v10 )
          return result;
      }
    }
  }
  return result;
}
