// Function: sub_2561E50
// Address: 0x2561e50
//
__int64 __fastcall sub_2561E50(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6, int a7)
{
  __int64 v7; // r15
  int v10; // edx
  unsigned int v11; // eax
  __int64 v12; // r13
  __int64 v13; // rcx
  unsigned __int64 v14; // rsi
  __m128i *v15; // r15
  __m128i *v16; // r12
  __m128i *v17; // rax
  char v18; // dl
  int v19; // eax
  __int64 result; // rax
  char v21; // dl
  __int64 v22; // rbx
  __int64 v23; // rdx
  __int64 v24; // rsi
  unsigned __int64 v25; // rax
  bool v26; // al
  unsigned __int64 v27; // rdx
  __m128i *v28; // rbx
  _QWORD *v29; // rax
  __int32 v30; // eax
  const void *v31; // rsi
  __int64 v32; // [rsp+8h] [rbp-68h]
  char *v34; // [rsp+18h] [rbp-58h]
  __m128i v35; // [rsp+20h] [rbp-50h] BYREF
  int v36; // [rsp+30h] [rbp-40h]

  v7 = 0xFFFFFFFFLL;
  v10 = a7;
  v34 = (char *)a6;
  if ( a3 )
  {
    _BitScanReverse(&v11, a3);
    a6 = 31 - (v11 ^ 0x1F);
    v7 = (int)a6;
  }
  v12 = *(_QWORD *)(a1 + 8 * v7 + 104);
  if ( !v12 )
  {
    v32 = a5;
    v29 = (_QWORD *)sub_A777F0(0x80u, *(__int64 **)(a1 + 168));
    a5 = v32;
    v10 = a7;
    v12 = (__int64)v29;
    if ( v29 )
    {
      memset(v29, 0, 0x80u);
      *v29 = v29 + 2;
      v29[1] = 0x200000000LL;
      v29[13] = v29 + 11;
      v29[14] = v29 + 11;
    }
    *(_QWORD *)(a1 + 8 * v7 + 104) = v29;
  }
  v13 = *(_QWORD *)(v12 + 120);
  v35.m128i_i64[0] = a4;
  v35.m128i_i64[1] = a5;
  v36 = v10;
  if ( v13 )
  {
    sub_253B210(v12 + 64, &v35);
    v18 = *v34 | v21;
    goto LABEL_13;
  }
  v14 = *(unsigned int *)(v12 + 8);
  v15 = *(__m128i **)v12;
  v16 = (__m128i *)(*(_QWORD *)v12 + 24 * v14);
  if ( *(__m128i **)v12 == v16 )
  {
    v22 = v12 + 64;
    if ( v14 <= 1 )
      goto LABEL_30;
LABEL_35:
    *(_DWORD *)(v12 + 8) = 0;
    sub_253B210(v22, &v35);
    v18 = 1;
    goto LABEL_13;
  }
  v17 = *(__m128i **)v12;
  do
  {
    if ( a4 == v17->m128i_i64[0] && a5 == v17->m128i_i64[1] && v10 == v17[1].m128i_i32[0] )
    {
      if ( v16 != v17 )
      {
        v18 = *v34;
        goto LABEL_13;
      }
      v22 = v12 + 64;
      if ( v14 <= 1 )
        goto LABEL_30;
LABEL_18:
      while ( v13 )
      {
        v23 = *(_QWORD *)(v12 + 112);
        if ( *(_QWORD *)(v23 + 32) == v15->m128i_i64[0] )
        {
          v25 = v15->m128i_u64[1];
          if ( *(_QWORD *)(v23 + 40) == v25 )
          {
            v30 = v15[1].m128i_i32[0];
            if ( *(_DWORD *)(v23 + 48) == v30 )
              break;
            v26 = *(_DWORD *)(v23 + 48) < v30;
          }
          else
          {
            v26 = *(_QWORD *)(v23 + 40) < v25;
          }
          if ( !v26 )
            break;
        }
        else if ( *(_QWORD *)(v23 + 32) >= v15->m128i_i64[0] )
        {
          break;
        }
        v24 = 0;
LABEL_26:
        sub_253B140(v22, v24, v23, v15);
LABEL_27:
        v15 = (__m128i *)((char *)v15 + 24);
        if ( v16 == v15 )
          goto LABEL_35;
        v13 = *(_QWORD *)(v12 + 120);
      }
      v24 = sub_253A680(v22, (unsigned __int64 *)v15);
      if ( !v23 )
        goto LABEL_27;
      goto LABEL_26;
    }
    v17 = (__m128i *)((char *)v17 + 24);
  }
  while ( v16 != v17 );
  v22 = v12 + 64;
  if ( v14 > 1 )
    goto LABEL_18;
LABEL_30:
  v27 = v14 + 1;
  v28 = &v35;
  if ( v14 + 1 > *(unsigned int *)(v12 + 12) )
  {
    v31 = (const void *)(v12 + 16);
    if ( v15 > &v35 || v16 <= &v35 )
    {
      sub_C8D5F0(v12, v31, v27, 0x18u, a5, a6);
      v16 = (__m128i *)(*(_QWORD *)v12 + 24LL * *(unsigned int *)(v12 + 8));
    }
    else
    {
      sub_C8D5F0(v12, v31, v27, 0x18u, a5, a6);
      v28 = (__m128i *)(*(_QWORD *)v12 + (char *)&v35 - (char *)v15);
      v16 = (__m128i *)(*(_QWORD *)v12 + 24LL * *(unsigned int *)(v12 + 8));
    }
  }
  v18 = 1;
  *v16 = _mm_loadu_si128(v28);
  v16[1].m128i_i64[0] = v28[1].m128i_i64[0];
  ++*(_DWORD *)(v12 + 8);
LABEL_13:
  *v34 = v18;
  v19 = ~a3;
  if ( a3 == 128 )
    v19 = -256;
  result = *(_DWORD *)(a2 + 8) | *(_DWORD *)(a2 + 12) & (unsigned int)v19;
  *(_DWORD *)(a2 + 12) = result;
  return result;
}
