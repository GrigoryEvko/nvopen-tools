// Function: sub_2E672B0
// Address: 0x2e672b0
//
void __fastcall sub_2E672B0(__m128i *a1, __m128i *a2, const __m128i *a3, __int64 a4)
{
  __m128i v4; // xmm0
  __m128i *v5; // rbx
  __m128i *v6; // r12
  const __m128i *v7; // rbx
  __int64 v8; // rsi
  __int64 v9; // r9
  char v10; // dl
  const __m128i *v11; // rcx
  int v12; // esi
  int v13; // r8d
  __int64 *v14; // r14
  unsigned int i; // edi
  __int64 *v16; // rax
  __int64 v17; // r10
  unsigned int v18; // edi
  unsigned int v19; // esi
  _DWORD *v20; // r14
  __int64 v21; // r15
  unsigned __int64 v22; // r13
  const __m128i *v23; // rdi
  int v24; // esi
  int v25; // r11d
  __int8 *v26; // rcx
  int v27; // eax
  __int8 *v28; // r8
  __int64 v29; // r10
  int v30; // eax
  unsigned int v31; // esi
  int v32; // edx
  int v33; // eax
  __m128i v34; // xmm1
  unsigned __int32 v35; // eax
  int v36; // ecx
  unsigned int v37; // edi
  unsigned __int32 v38; // eax
  int v39; // edi
  unsigned int v40; // r8d
  const __m128i *v41; // rsi
  int v42; // edx
  int v43; // r10d
  __int8 *v44; // r8
  int m; // eax
  __int64 v46; // rdi
  int v47; // eax
  const __m128i *v48; // rcx
  int v49; // eax
  int v50; // r8d
  __int64 *v51; // rdi
  unsigned int j; // esi
  __int64 v53; // rdx
  unsigned int v54; // esi
  const __m128i *v55; // rsi
  int v56; // edx
  int v57; // r10d
  int n; // eax
  __int64 v59; // rdi
  int v60; // eax
  const __m128i *v61; // rcx
  int v62; // eax
  int v63; // r8d
  unsigned int k; // esi
  __int64 v65; // rdx
  unsigned int v66; // esi
  __int32 v67; // eax
  __int32 v68; // edx
  __int32 v69; // edx
  __int32 v70; // eax
  __int64 v71; // [rsp+0h] [rbp-90h]
  __int64 v72; // [rsp+0h] [rbp-90h]
  __int64 v73; // [rsp+0h] [rbp-90h]
  __int64 v74; // [rsp+0h] [rbp-90h]
  __int64 v75; // [rsp+8h] [rbp-88h]
  __m128i *v78; // [rsp+28h] [rbp-68h]
  unsigned __int64 v79; // [rsp+30h] [rbp-60h]
  int v80; // [rsp+3Ch] [rbp-54h]
  __m128i v81; // [rsp+40h] [rbp-50h] BYREF
  const __m128i *v82; // [rsp+50h] [rbp-40h] BYREF
  __int64 v83; // [rsp+58h] [rbp-38h]

  v82 = a3;
  v83 = a4;
  if ( a1 == a2 || a2 == &a1[1] )
    return;
  v78 = a1 + 1;
  do
  {
    while ( sub_2E651A0(&v82, v78->m128i_i64, a1->m128i_i64) )
    {
      v4 = _mm_loadu_si128(v78);
      v5 = v78 + 1;
      if ( a1 != v78 )
      {
        v81 = v4;
        memmove(&a1[1], a1, (char *)v78 - (char *)a1);
        v4 = _mm_load_si128(&v81);
      }
      ++v78;
      *a1 = v4;
      if ( a2 == v5 )
        return;
    }
    v6 = v78;
    v7 = v82;
    v8 = v78->m128i_i64[1];
    v9 = v78->m128i_i64[0];
    v81.m128i_i64[0] = v83;
    v75 = v8;
    v79 = v8 & 0xFFFFFFFFFFFFFFF8LL;
    v80 = ((0xBF58476D1CE4E5B9LL
          * (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)
           | ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32))) >> 31)
        ^ (484763065 * (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)));
    while ( 2 )
    {
      v10 = v7->m128i_i8[8] & 1;
      if ( v10 )
      {
        v11 = v7 + 1;
        v12 = 3;
      }
      else
      {
        v19 = v7[1].m128i_u32[2];
        v11 = (const __m128i *)v7[1].m128i_i64[0];
        if ( !v19 )
        {
          v35 = v7->m128i_u32[2];
          ++v7->m128i_i64[0];
          v14 = 0;
          v36 = (v35 >> 1) + 1;
          goto LABEL_44;
        }
        v12 = v19 - 1;
      }
      v13 = 1;
      v14 = 0;
      for ( i = v12 & v80; ; i = v12 & v18 )
      {
        v16 = &v11->m128i_i64[3 * i];
        v17 = *v16;
        if ( v9 == *v16 && v16[1] == v79 )
        {
          v20 = v16 + 2;
          goto LABEL_23;
        }
        if ( v17 == -4096 )
          break;
        if ( v17 == -8192 && v16[1] == -8192 && !v14 )
          v14 = &v11->m128i_i64[3 * i];
LABEL_18:
        v18 = v13 + i;
        ++v13;
      }
      if ( v16[1] != -4096 )
        goto LABEL_18;
      v37 = 12;
      v19 = 4;
      if ( !v14 )
        v14 = v16;
      v35 = v7->m128i_u32[2];
      ++v7->m128i_i64[0];
      v36 = (v35 >> 1) + 1;
      if ( !v10 )
      {
        v19 = v7[1].m128i_u32[2];
LABEL_44:
        v37 = 3 * v19;
      }
      if ( 4 * v36 >= v37 )
      {
        v72 = v9;
        sub_2E64C00(v7, 2 * v19);
        v9 = v72;
        if ( (v7->m128i_i8[8] & 1) != 0 )
        {
          v48 = v7 + 1;
          v49 = 3;
        }
        else
        {
          v67 = v7[1].m128i_i32[2];
          v48 = (const __m128i *)v7[1].m128i_i64[0];
          if ( !v67 )
            goto LABEL_137;
          v49 = v67 - 1;
        }
        v50 = 1;
        v51 = 0;
        for ( j = v49 & v80; ; j = v49 & v54 )
        {
          v14 = &v48->m128i_i64[3 * j];
          v53 = *v14;
          if ( v72 == *v14 && v14[1] == v79 )
            break;
          if ( v53 == -4096 )
          {
            if ( v14[1] == -4096 )
            {
LABEL_131:
              if ( v51 )
                v14 = v51;
              goto LABEL_122;
            }
          }
          else if ( v53 == -8192 && v14[1] == -8192 && !v51 )
          {
            v51 = &v48->m128i_i64[3 * j];
          }
          v54 = v50 + j;
          ++v50;
        }
        goto LABEL_122;
      }
      if ( v19 - v7->m128i_i32[3] - v36 <= v19 >> 3 )
      {
        v74 = v9;
        sub_2E64C00(v7, v19);
        v9 = v74;
        if ( (v7->m128i_i8[8] & 1) != 0 )
        {
          v61 = v7 + 1;
          v62 = 3;
        }
        else
        {
          v70 = v7[1].m128i_i32[2];
          v61 = (const __m128i *)v7[1].m128i_i64[0];
          if ( !v70 )
            goto LABEL_137;
          v62 = v70 - 1;
        }
        v63 = 1;
        v51 = 0;
        for ( k = v62 & v80; ; k = v62 & v66 )
        {
          v14 = &v61->m128i_i64[3 * k];
          v65 = *v14;
          if ( v74 == *v14 && v14[1] == v79 )
            break;
          if ( v65 == -4096 )
          {
            if ( v14[1] == -4096 )
              goto LABEL_131;
          }
          else if ( v65 == -8192 && v14[1] == -8192 && !v51 )
          {
            v51 = &v61->m128i_i64[3 * k];
          }
          v66 = v63 + k;
          ++v63;
        }
LABEL_122:
        v35 = v7->m128i_u32[2];
      }
      v7->m128i_i32[2] = (2 * (v35 >> 1) + 2) | v35 & 1;
      if ( *v14 != -4096 || v14[1] != -4096 )
        --v7->m128i_i32[3];
      *v14 = v9;
      v20 = v14 + 2;
      *v20 = 0;
      *((_QWORD *)v20 - 1) = v79;
      v10 = v7->m128i_i8[8] & 1;
LABEL_23:
      v21 = v6[-1].m128i_i64[0];
      v22 = v6[-1].m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL;
      if ( v10 )
      {
        v23 = v7 + 1;
        v24 = 3;
      }
      else
      {
        v31 = v7[1].m128i_u32[2];
        v23 = (const __m128i *)v7[1].m128i_i64[0];
        if ( !v31 )
        {
          v38 = v7->m128i_u32[2];
          ++v7->m128i_i64[0];
          v26 = 0;
          v39 = (v38 >> 1) + 1;
          goto LABEL_51;
        }
        v24 = v31 - 1;
      }
      v25 = 1;
      v26 = 0;
      v27 = v24
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4)
              | ((unsigned __int64)(((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4))));
      while ( 2 )
      {
        v28 = &v23->m128i_i8[24 * v27];
        v29 = *(_QWORD *)v28;
        if ( v21 == *(_QWORD *)v28 && v22 == *((_QWORD *)v28 + 1) )
        {
          v32 = *((_DWORD *)v28 + 4);
          goto LABEL_37;
        }
        if ( v29 != -4096 )
        {
          if ( v29 == -8192 && *((_QWORD *)v28 + 1) == -8192 && !v26 )
            v26 = &v23->m128i_i8[24 * v27];
          goto LABEL_32;
        }
        if ( *((_QWORD *)v28 + 1) != -4096 )
        {
LABEL_32:
          v30 = v25 + v27;
          ++v25;
          v27 = v24 & v30;
          continue;
        }
        break;
      }
      v38 = v7->m128i_u32[2];
      v31 = 4;
      if ( !v26 )
        v26 = v28;
      ++v7->m128i_i64[0];
      v40 = 12;
      v39 = (v38 >> 1) + 1;
      if ( !v10 )
      {
        v31 = v7[1].m128i_u32[2];
LABEL_51:
        v40 = 3 * v31;
      }
      if ( 4 * v39 >= v40 )
      {
        v71 = v9;
        sub_2E64C00(v7, 2 * v31);
        v9 = v71;
        if ( (v7->m128i_i8[8] & 1) != 0 )
        {
          v41 = v7 + 1;
          v42 = 3;
LABEL_63:
          v43 = 1;
          v44 = 0;
          for ( m = v42
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4)
                      | ((unsigned __int64)(((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4)))); ; m = v42 & v47 )
          {
            v26 = &v41->m128i_i8[24 * m];
            v46 = *(_QWORD *)v26;
            if ( v21 == *(_QWORD *)v26 && v22 == *((_QWORD *)v26 + 1) )
              break;
            if ( v46 == -4096 )
            {
              if ( *((_QWORD *)v26 + 1) == -4096 )
              {
LABEL_128:
                if ( v44 )
                  v26 = v44;
                break;
              }
            }
            else if ( v46 == -8192 && *((_QWORD *)v26 + 1) == -8192 && !v44 )
            {
              v44 = &v41->m128i_i8[24 * m];
            }
            v47 = v43 + m;
            ++v43;
          }
LABEL_120:
          v38 = v7->m128i_u32[2];
          goto LABEL_54;
        }
        v68 = v7[1].m128i_i32[2];
        v41 = (const __m128i *)v7[1].m128i_i64[0];
        if ( v68 )
        {
          v42 = v68 - 1;
          goto LABEL_63;
        }
LABEL_137:
        v7->m128i_i32[2] = (2 * ((unsigned __int32)v7->m128i_i32[2] >> 1) + 2) | v7->m128i_i32[2] & 1;
        BUG();
      }
      if ( v31 - v7->m128i_i32[3] - v39 <= v31 >> 3 )
      {
        v73 = v9;
        sub_2E64C00(v7, v31);
        v9 = v73;
        if ( (v7->m128i_i8[8] & 1) != 0 )
        {
          v55 = v7 + 1;
          v56 = 3;
        }
        else
        {
          v69 = v7[1].m128i_i32[2];
          v55 = (const __m128i *)v7[1].m128i_i64[0];
          if ( !v69 )
            goto LABEL_137;
          v56 = v69 - 1;
        }
        v57 = 1;
        v44 = 0;
        for ( n = v56
                & (((0xBF58476D1CE4E5B9LL
                   * (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4)
                    | ((unsigned __int64)(((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4)) << 32))) >> 31)
                 ^ (484763065 * (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4)))); ; n = v56 & v60 )
        {
          v26 = &v55->m128i_i8[24 * n];
          v59 = *(_QWORD *)v26;
          if ( v21 == *(_QWORD *)v26 && v22 == *((_QWORD *)v26 + 1) )
            break;
          if ( v59 == -4096 )
          {
            if ( *((_QWORD *)v26 + 1) == -4096 )
              goto LABEL_128;
          }
          else if ( v59 == -8192 && *((_QWORD *)v26 + 1) == -8192 && !v44 )
          {
            v44 = &v55->m128i_i8[24 * n];
          }
          v60 = v57 + n;
          ++v57;
        }
        goto LABEL_120;
      }
LABEL_54:
      v7->m128i_i32[2] = (2 * (v38 >> 1) + 2) | v38 & 1;
      if ( *(_QWORD *)v26 != -4096 || *((_QWORD *)v26 + 1) != -4096 )
        --v7->m128i_i32[3];
      *(_QWORD *)v26 = v21;
      v32 = 0;
      *((_QWORD *)v26 + 1) = v22;
      *((_DWORD *)v26 + 4) = 0;
LABEL_37:
      v33 = *v20;
      if ( !*(_BYTE *)v81.m128i_i64[0] )
      {
        if ( v32 >= v33 )
          goto LABEL_41;
LABEL_39:
        v34 = _mm_loadu_si128(--v6);
        v6[1] = v34;
        continue;
      }
      break;
    }
    if ( v32 > v33 )
      goto LABEL_39;
LABEL_41:
    v6->m128i_i64[0] = v9;
    v6->m128i_i64[1] = v75;
    ++v78;
  }
  while ( a2 != v78 );
}
