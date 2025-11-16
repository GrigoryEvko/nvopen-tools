// Function: sub_B23600
// Address: 0xb23600
//
void __fastcall sub_B23600(__m128i *a1, __m128i *a2, __int64 a3, __int64 a4)
{
  __m128i *v4; // r15
  __m128i v5; // xmm0
  __int64 v6; // r14
  __m128i *v7; // r13
  __int64 v8; // rdi
  __int64 v9; // rbx
  unsigned __int64 v10; // r15
  char v11; // al
  __int64 v12; // rdx
  int v13; // esi
  int v14; // r9d
  __int64 *v15; // r8
  unsigned int i; // edi
  __int64 *v17; // r12
  __int64 v18; // rcx
  unsigned int v19; // edi
  unsigned int v20; // esi
  __int64 v21; // rdi
  _DWORD *v22; // r12
  unsigned __int64 v23; // rcx
  char v24; // r8
  __int64 v25; // r9
  int v26; // esi
  __int64 *v27; // r11
  unsigned int j; // eax
  __int64 *v29; // rdx
  __int64 v30; // r10
  unsigned int v31; // eax
  unsigned int v32; // esi
  int v33; // eax
  __m128i v34; // xmm1
  unsigned int v35; // edx
  int v36; // ecx
  unsigned int v37; // edi
  __int64 v38; // rax
  unsigned int v39; // edx
  int v40; // ecx
  unsigned int v41; // edi
  __int64 v42; // rax
  __m128i *v43; // [rsp+0h] [rbp-A0h]
  __int64 v44; // [rsp+8h] [rbp-98h]
  int v47; // [rsp+28h] [rbp-78h]
  int v48; // [rsp+2Ch] [rbp-74h]
  __m128i v49; // [rsp+30h] [rbp-70h] BYREF
  __int64 v50; // [rsp+40h] [rbp-60h] BYREF
  __int64 v51; // [rsp+48h] [rbp-58h]
  __int64 *v52; // [rsp+58h] [rbp-48h] BYREF
  __int64 v53; // [rsp+60h] [rbp-40h] BYREF
  unsigned __int64 v54; // [rsp+68h] [rbp-38h]

  v50 = a3;
  v51 = a4;
  if ( a1 != a2 && a2 != &a1[1] )
  {
    v4 = a1 + 1;
    do
    {
      while ( sub_B1DED0((__int64)&v50, v4->m128i_i64, a1->m128i_i64) )
      {
        v5 = _mm_loadu_si128(v4);
        if ( a1 != v4 )
        {
          v49 = v5;
          memmove(&a1[1], a1, (char *)v4 - (char *)a1);
          v5 = _mm_load_si128(&v49);
        }
        ++v4;
        *a1 = v5;
        if ( a2 == v4 )
          return;
      }
      v6 = v4->m128i_i64[0];
      v7 = v4;
      v8 = v4->m128i_i64[1];
      v9 = v50;
      v43 = v4;
      v49.m128i_i64[0] = v51;
      v44 = v8;
      v10 = v8 & 0xFFFFFFFFFFFFFFF8LL;
      v48 = ((0xBF58476D1CE4E5B9LL
            * (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)
             | ((unsigned __int64)(((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)) << 32))) >> 31)
          ^ (484763065 * (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)));
      while ( 1 )
      {
        v53 = v6;
        v54 = v10;
        v11 = *(_BYTE *)(v9 + 8) & 1;
        if ( v11 )
        {
          v12 = v9 + 16;
          v13 = 3;
        }
        else
        {
          v20 = *(_DWORD *)(v9 + 24);
          v12 = *(_QWORD *)(v9 + 16);
          if ( !v20 )
          {
            v52 = 0;
            v35 = *(_DWORD *)(v9 + 8);
            ++*(_QWORD *)v9;
            v36 = (v35 >> 1) + 1;
LABEL_42:
            v37 = 3 * v20;
LABEL_43:
            if ( 4 * v36 >= v37 )
            {
              v20 *= 2;
            }
            else if ( v20 - *(_DWORD *)(v9 + 12) - v36 > v20 >> 3 )
            {
              goto LABEL_45;
            }
            sub_B1DB20(v9, v20);
            sub_B1C410(v9, &v53, &v52);
            v35 = *(_DWORD *)(v9 + 8);
LABEL_45:
            *(_DWORD *)(v9 + 8) = (2 * (v35 >> 1) + 2) | v35 & 1;
            v17 = v52;
            if ( *v52 != -4096 || v52[1] != -4096 )
              --*(_DWORD *)(v9 + 12);
            *v17 = v53;
            v38 = v54;
            *((_DWORD *)v17 + 4) = 0;
            v17[1] = v38;
            goto LABEL_22;
          }
          v13 = v20 - 1;
        }
        v14 = 1;
        v15 = 0;
        for ( i = v13 & v48; ; i = v13 & v19 )
        {
          v17 = (__int64 *)(v12 + 24LL * i);
          v18 = *v17;
          if ( v6 == *v17 && v17[1] == v10 )
            break;
          if ( v18 == -4096 )
          {
            if ( v17[1] == -4096 )
            {
              v37 = 12;
              v20 = 4;
              if ( !v15 )
                v15 = v17;
              v52 = v15;
              v35 = *(_DWORD *)(v9 + 8);
              ++*(_QWORD *)v9;
              v36 = (v35 >> 1) + 1;
              if ( !v11 )
              {
                v20 = *(_DWORD *)(v9 + 24);
                goto LABEL_42;
              }
              goto LABEL_43;
            }
          }
          else if ( v18 == -8192 && v17[1] == -8192 && !v15 )
          {
            v15 = (__int64 *)(v12 + 24LL * i);
          }
          v19 = v14 + i;
          ++v14;
        }
LABEL_22:
        v21 = v7[-1].m128i_i64[0];
        v22 = v17 + 2;
        v23 = v7[-1].m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL;
        v53 = v21;
        v54 = v23;
        v24 = *(_BYTE *)(v9 + 8) & 1;
        if ( v24 )
        {
          v25 = v9 + 16;
          v26 = 3;
        }
        else
        {
          v32 = *(_DWORD *)(v9 + 24);
          v25 = *(_QWORD *)(v9 + 16);
          if ( !v32 )
          {
            v52 = 0;
            v39 = *(_DWORD *)(v9 + 8);
            ++*(_QWORD *)v9;
            v40 = (v39 >> 1) + 1;
LABEL_49:
            v41 = 3 * v32;
LABEL_50:
            if ( 4 * v40 >= v41 )
            {
              v32 *= 2;
            }
            else if ( v32 - *(_DWORD *)(v9 + 12) - v40 > v32 >> 3 )
            {
              goto LABEL_52;
            }
            sub_B1DB20(v9, v32);
            sub_B1C410(v9, &v53, &v52);
            v39 = *(_DWORD *)(v9 + 8);
LABEL_52:
            *(_DWORD *)(v9 + 8) = (2 * (v39 >> 1) + 2) | v39 & 1;
            v29 = v52;
            if ( *v52 != -4096 || v52[1] != -4096 )
              --*(_DWORD *)(v9 + 12);
            *v29 = v53;
            v42 = v54;
            *((_DWORD *)v29 + 4) = 0;
            v29[1] = v42;
            goto LABEL_35;
          }
          v26 = v32 - 1;
        }
        v47 = 1;
        v27 = 0;
        for ( j = v26
                & (((0xBF58476D1CE4E5B9LL
                   * (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4)
                    | ((unsigned __int64)(((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4)) << 32))) >> 31)
                 ^ (484763065 * (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4)))); ; j = v26 & v31 )
        {
          v29 = (__int64 *)(v25 + 24LL * j);
          v30 = *v29;
          if ( v21 == *v29 && v23 == v29[1] )
            break;
          if ( v30 == -4096 )
          {
            if ( v29[1] == -4096 )
            {
              v41 = 12;
              v32 = 4;
              if ( !v27 )
                v27 = (__int64 *)(v25 + 24LL * j);
              v52 = v27;
              v39 = *(_DWORD *)(v9 + 8);
              ++*(_QWORD *)v9;
              v40 = (v39 >> 1) + 1;
              if ( !v24 )
              {
                v32 = *(_DWORD *)(v9 + 24);
                goto LABEL_49;
              }
              goto LABEL_50;
            }
          }
          else if ( v30 == -8192 && v29[1] == -8192 && !v27 )
          {
            v27 = (__int64 *)(v25 + 24LL * j);
          }
          v31 = v47 + j;
          ++v47;
        }
LABEL_35:
        v33 = *((_DWORD *)v29 + 4);
        if ( !*(_BYTE *)v49.m128i_i64[0] )
          break;
        if ( *v22 >= v33 )
          goto LABEL_39;
LABEL_37:
        v34 = _mm_loadu_si128(--v7);
        v7[1] = v34;
      }
      if ( *v22 > v33 )
        goto LABEL_37;
LABEL_39:
      v7->m128i_i64[0] = v6;
      v7->m128i_i64[1] = v44;
      v4 = v43 + 1;
    }
    while ( a2 != &v43[1] );
  }
}
