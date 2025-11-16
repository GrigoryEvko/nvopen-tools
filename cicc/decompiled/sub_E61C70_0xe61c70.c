// Function: sub_E61C70
// Address: 0xe61c70
//
__int64 __fastcall sub_E61C70(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rdx
  unsigned __int64 v8; // r15
  unsigned __int64 v9; // rbx
  _DWORD *v10; // r9
  unsigned __int64 v11; // rax
  _DWORD *v13; // r14
  unsigned __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // rsi
  unsigned int v17; // edi
  int *v18; // rax
  int v19; // r10d
  unsigned int v20; // edx
  int v21; // ecx
  __m128i *v22; // rsi
  int v23; // eax
  __int64 v24; // rdi
  __int64 v25; // rdx
  const __m128i *v26; // r8
  __int32 v27; // ecx
  __int64 v28; // rdi
  int v29; // eax
  unsigned __int64 v30; // [rsp+0h] [rbp-60h]
  unsigned __int64 v31; // [rsp+0h] [rbp-60h]
  int v32; // [rsp+0h] [rbp-60h]
  __m128i v33; // [rsp+10h] [rbp-50h] BYREF
  __int64 v34; // [rsp+20h] [rbp-40h]

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  v5 = sub_E5FF70(a2, a3);
  if ( v6 > v5 )
  {
    v8 = v6;
    v9 = v5;
    v10 = sub_E5F790(a2, a3);
    v11 = v8;
    v13 = v10;
    v14 = v11;
    while ( 1 )
    {
      v25 = *(_QWORD *)(a2 + 256);
      v26 = (const __m128i *)(v25 + 24 * v9);
      v27 = v26->m128i_i32[2];
      if ( a3 == v27 )
        break;
      v15 = (unsigned int)v13[12];
      v16 = *((_QWORD *)v13 + 4);
      if ( (_DWORD)v15 )
      {
        v17 = (v15 - 1) & (37 * v27);
        v18 = (int *)(v16 + 16LL * v17);
        v19 = *v18;
        if ( v27 != *v18 )
        {
          v29 = 1;
          while ( v19 != -1 )
          {
            v17 = (v15 - 1) & (v29 + v17);
            v32 = v29 + 1;
            v18 = (int *)(v16 + 16LL * v17);
            v19 = *v18;
            if ( v27 == *v18 )
              goto LABEL_6;
            v29 = v32;
          }
          goto LABEL_12;
        }
LABEL_6:
        if ( v18 != (int *)(v16 + 16 * v15) )
        {
          v20 = v18[1];
          v21 = v18[2];
          v22 = *(__m128i **)(a1 + 8);
          v23 = v18[3];
          if ( v22 != *(__m128i **)a1 )
          {
            if ( v22[-1].m128i_i32[1] == v20 && v21 == v22[-1].m128i_i32[2] && v22[-1].m128i_u16[6] == v23 )
              goto LABEL_12;
            v24 = v26->m128i_i64[0];
            BYTE6(v34) &= 0xFCu;
            v33.m128i_i64[1] = __PAIR64__(v20, a3);
            v33.m128i_i64[0] = v24;
            LODWORD(v34) = v21;
            WORD2(v34) = v23;
            if ( *(__m128i **)(a1 + 16) == v22 )
            {
LABEL_20:
              v30 = v14;
              sub_E61AC0((const __m128i **)a1, v22, &v33);
              v14 = v30;
              goto LABEL_12;
            }
            goto LABEL_10;
          }
          v28 = v26->m128i_i64[0];
          BYTE6(v34) &= 0xFCu;
          v33.m128i_i64[1] = __PAIR64__(v20, a3);
          v33.m128i_i64[0] = v28;
          LODWORD(v34) = v21;
          WORD2(v34) = v23;
          if ( v22 == *(__m128i **)(a1 + 16) )
            goto LABEL_20;
          if ( v22 )
          {
LABEL_10:
            *v22 = _mm_loadu_si128(&v33);
            v22[1].m128i_i64[0] = v34;
            v22 = *(__m128i **)(a1 + 8);
          }
LABEL_11:
          *(_QWORD *)(a1 + 8) = (char *)v22 + 24;
        }
      }
LABEL_12:
      if ( v14 == ++v9 )
        return a1;
    }
    v22 = *(__m128i **)(a1 + 8);
    if ( v22 == *(__m128i **)(a1 + 16) )
    {
      v31 = v14;
      sub_E61180((const __m128i **)a1, v22, (const __m128i *)(v25 + 24 * v9));
      v14 = v31;
      goto LABEL_12;
    }
    if ( v22 )
    {
      *v22 = _mm_loadu_si128(v26);
      v22[1].m128i_i64[0] = v26[1].m128i_i64[0];
      v22 = *(__m128i **)(a1 + 8);
    }
    goto LABEL_11;
  }
  return a1;
}
