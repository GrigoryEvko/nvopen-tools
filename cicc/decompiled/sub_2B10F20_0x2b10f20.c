// Function: sub_2B10F20
// Address: 0x2b10f20
//
__m128i *__fastcall sub_2B10F20(__m128i **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __m128i *result; // rax
  __int64 v7; // r15
  int v10; // r12d
  __int64 v11; // rbx
  const void *v12; // rsi
  __m128i *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __m128i *v16; // rcx
  __int32 v17; // edi
  __m128i *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __m128i *v21; // rcx
  __int32 v22; // edi
  __m128i v23; // xmm0
  __int64 v24; // rax
  int v25; // ecx
  int v26; // r11d
  int v27; // ecx
  int v28; // r11d
  __m128i v29; // [rsp+0h] [rbp-50h] BYREF
  const void *v30; // [rsp+18h] [rbp-38h]

  result = *a1;
  v7 = (*a1)->m128i_u32[2];
  if ( (*a1)->m128i_i32[2] )
  {
    v10 = 0;
    v11 = 0;
    v12 = (const void *)(a2 + 16);
    while ( 1 )
    {
      v13 = a1[1];
      v14 = v13->m128i_i64[1];
      v15 = v13[1].m128i_u32[2];
      if ( (_DWORD)v15 )
      {
        a6 = (unsigned int)(v15 - 1);
        a5 = v10 & (unsigned int)(v15 - 1);
        v16 = (__m128i *)(v14 + 4 * a5);
        v17 = v16->m128i_i32[0];
        if ( (_DWORD)v11 == v16->m128i_i32[0] )
        {
LABEL_6:
          result = (__m128i *)(v14 + 4 * v15);
          if ( v16 != result )
            goto LABEL_3;
        }
        else
        {
          v25 = 1;
          while ( v17 != -1 )
          {
            v26 = v25 + 1;
            a5 = (unsigned int)a6 & (v25 + (_DWORD)a5);
            v16 = (__m128i *)(v14 + 4LL * (unsigned int)a5);
            v17 = v16->m128i_i32[0];
            if ( v16->m128i_i32[0] == (_DWORD)v11 )
              goto LABEL_6;
            v25 = v26;
          }
        }
      }
      v18 = a1[2];
      v19 = v18->m128i_i64[1];
      v20 = v18[1].m128i_u32[2];
      if ( !(_DWORD)v20 )
        goto LABEL_10;
      a6 = (unsigned int)(v20 - 1);
      a5 = v10 & (unsigned int)(v20 - 1);
      v21 = (__m128i *)(v19 + 4 * a5);
      v22 = v21->m128i_i32[0];
      if ( (_DWORD)v11 != v21->m128i_i32[0] )
      {
        v27 = 1;
        while ( v22 != -1 )
        {
          v28 = v27 + 1;
          a5 = (unsigned int)a6 & (v27 + (_DWORD)a5);
          v21 = (__m128i *)(v19 + 4LL * (unsigned int)a5);
          v22 = v21->m128i_i32[0];
          if ( (_DWORD)v11 == v21->m128i_i32[0] )
            goto LABEL_9;
          v27 = v28;
        }
        goto LABEL_10;
      }
LABEL_9:
      result = (__m128i *)(v19 + 4 * v20);
      if ( v21 != result )
      {
LABEL_3:
        ++v11;
        v10 += 37;
        if ( v7 == v11 )
          return result;
      }
      else
      {
LABEL_10:
        v23 = _mm_loadu_si128((const __m128i *)((*a1)->m128i_i64[0] + 16LL * (unsigned int)v11));
        v24 = *(unsigned int *)(a2 + 8);
        if ( v24 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        {
          v30 = v12;
          v29 = v23;
          sub_C8D5F0(a2, v12, v24 + 1, 0x10u, a5, a6);
          v24 = *(unsigned int *)(a2 + 8);
          v23 = _mm_load_si128(&v29);
          v12 = v30;
        }
        ++v11;
        result = (__m128i *)(*(_QWORD *)a2 + 16 * v24);
        v10 += 37;
        *result = v23;
        ++*(_DWORD *)(a2 + 8);
        if ( v7 == v11 )
          return result;
      }
    }
  }
  return result;
}
