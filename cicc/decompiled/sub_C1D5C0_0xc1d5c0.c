// Function: sub_C1D5C0
// Address: 0xc1d5c0
//
__int64 __fastcall sub_C1D5C0(__m128i *a1, const __m128i *a2, unsigned __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  bool v6; // zf
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rdi
  __int64 v9; // rax
  unsigned int v10; // r12d
  __int64 v11; // r14
  __m128i *v12; // rbx
  __int64 v13; // rax
  unsigned int v14; // edx
  __int64 m128i_i64; // r15
  unsigned int v16; // eax
  __int64 v17; // r12
  _QWORD *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned __int32 v21; // edx
  __m128i *v22; // rcx
  _QWORD *v23; // rbx
  _QWORD *v24; // r15
  size_t v25; // r14
  const void *v26; // r13
  size_t v27; // rdx
  int v28; // eax
  size_t v29; // r12
  const void *v30; // rdi
  size_t v31; // rcx
  const void *v32; // rsi
  size_t v33; // rdx
  int v34; // eax
  _QWORD *v35; // r13
  unsigned int v36; // eax
  __int64 v38; // rax
  __int64 v39; // rax
  __m128i *v40; // rdx
  _BOOL8 v41; // rdi
  _QWORD *v42; // rsi
  __int64 v43; // r15
  _QWORD *v44; // rax
  _QWORD *v45; // rdx
  _QWORD *v46; // r9
  _BOOL8 v47; // rdi
  unsigned __int64 v48; // rbx
  unsigned __int64 v49; // r14
  const void *v50; // rsi
  const void *v51; // rdi
  size_t v52; // rdx
  unsigned int v53; // eax
  unsigned int v54; // eax
  const __m128i *v55; // [rsp+10h] [rbp-B0h]
  size_t v56; // [rsp+20h] [rbp-A0h]
  _QWORD *v57; // [rsp+20h] [rbp-A0h]
  __int64 v58; // [rsp+38h] [rbp-88h]
  _QWORD *v61; // [rsp+58h] [rbp-68h]
  unsigned int v62; // [rsp+64h] [rbp-5Ch]
  _QWORD *v63; // [rsp+68h] [rbp-58h]
  const __m128i *v64; // [rsp+70h] [rbp-50h]
  __int64 v65; // [rsp+70h] [rbp-50h]
  __int64 v66; // [rsp+70h] [rbp-50h]
  __int64 *v67[7]; // [rsp+88h] [rbp-38h] BYREF

  if ( !a1->m128i_i64[0] )
    a1->m128i_i64[0] = a2->m128i_i64[0];
  if ( !a1[1].m128i_i64[1] )
  {
    a1[1] = _mm_loadu_si128(a2 + 1);
    a1[2] = _mm_loadu_si128(a2 + 2);
    a1[3].m128i_i64[0] = a2[3].m128i_i64[0];
  }
  v3 = a1->m128i_i64[1];
  v4 = a2->m128i_i64[1];
  if ( v3 )
  {
    v62 = 14;
    if ( v3 != v4 )
      return v62;
  }
  else
  {
    a1->m128i_i64[1] = v4;
  }
  v5 = sub_C1B1E0(a2[3].m128i_u64[1], a3, a1[3].m128i_u64[1], (bool *)v67);
  v6 = LOBYTE(v67[0]) == 0;
  a1[3].m128i_i64[1] = v5;
  v7 = a1[4].m128i_u64[0];
  v8 = a2[4].m128i_u64[0];
  if ( !v6 )
  {
    a1[4].m128i_i64[0] = sub_C1B1E0(v8, a3, v7, (bool *)v67);
    goto LABEL_74;
  }
  v9 = sub_C1B1E0(v8, a3, v7, (bool *)v67);
  v6 = LOBYTE(v67[0]) == 0;
  a1[4].m128i_i64[0] = v9;
  if ( !v6 )
  {
LABEL_74:
    v62 = 10;
    goto LABEL_10;
  }
  v62 = 0;
LABEL_10:
  if ( (const __m128i *)a2[6].m128i_i64[0] == &a2[5] )
    goto LABEL_26;
  v10 = v62;
  v11 = a2[6].m128i_i64[0];
  v12 = a1 + 5;
  do
  {
    v13 = a1[5].m128i_i64[1];
    if ( v13 )
    {
      v14 = *(_DWORD *)(v11 + 32);
      m128i_i64 = (__int64)a1[5].m128i_i64;
      while ( 1 )
      {
        while ( *(_DWORD *)(v13 + 32) < v14 )
        {
          v13 = *(_QWORD *)(v13 + 24);
LABEL_18:
          if ( !v13 )
          {
LABEL_19:
            if ( (__m128i *)m128i_i64 == v12
              || *(_DWORD *)(m128i_i64 + 32) > v14
              || *(_DWORD *)(m128i_i64 + 32) == v14 && *(_DWORD *)(v11 + 36) < *(_DWORD *)(m128i_i64 + 36) )
            {
              goto LABEL_78;
            }
            goto LABEL_22;
          }
        }
        if ( *(_DWORD *)(v13 + 32) == v14 && *(_DWORD *)(v13 + 36) < *(_DWORD *)(v11 + 36) )
        {
          v13 = *(_QWORD *)(v13 + 24);
          goto LABEL_18;
        }
        m128i_i64 = v13;
        v13 = *(_QWORD *)(v13 + 16);
        if ( !v13 )
          goto LABEL_19;
      }
    }
    m128i_i64 = (__int64)a1[5].m128i_i64;
LABEL_78:
    v65 = m128i_i64;
    m128i_i64 = sub_22077B0(104);
    v38 = *(_QWORD *)(v11 + 32);
    *(_OWORD *)(m128i_i64 + 40) = 0;
    *(_QWORD *)(m128i_i64 + 32) = v38;
    *(_OWORD *)(m128i_i64 + 72) = 0;
    *(_QWORD *)(m128i_i64 + 48) = m128i_i64 + 96;
    *(_QWORD *)(m128i_i64 + 56) = 1;
    *(_QWORD *)(m128i_i64 + 64) = 0;
    *(_DWORD *)(m128i_i64 + 80) = 1065353216;
    *(_OWORD *)(m128i_i64 + 88) = 0;
    v39 = sub_C1D150(&a1[4].m128i_i64[1], v65, (unsigned int *)(m128i_i64 + 32));
    if ( v40 )
    {
      v41 = 1;
      if ( v12 != v40 && !v39 )
      {
        v54 = v40[2].m128i_u32[0];
        if ( *(_DWORD *)(m128i_i64 + 32) >= v54 )
        {
          v41 = 0;
          if ( *(_DWORD *)(m128i_i64 + 32) == v54 )
            v41 = *(_DWORD *)(m128i_i64 + 36) < v40[2].m128i_i32[1];
        }
      }
      sub_220F040(v41, m128i_i64, v40, v12);
      ++a1[7].m128i_i64[0];
    }
    else
    {
      v66 = v39;
      j_j___libc_free_0(m128i_i64, 104);
      m128i_i64 = v66;
    }
LABEL_22:
    v16 = sub_C1CFB0((unsigned __int64 *)(m128i_i64 + 40), (unsigned __int64 *)(v11 + 40), a3);
    if ( !v10 )
      v10 = v16;
    v11 = sub_220EF30(v11);
  }
  while ( &a2[5] != (const __m128i *)v11 );
  v62 = v10;
LABEL_26:
  v55 = a2 + 8;
  v58 = a2[9].m128i_i64[0];
  if ( &a2[8] != (const __m128i *)v58 )
  {
LABEL_27:
    v17 = v58 + 32;
    v18 = (_QWORD *)a1[10].m128i_i64[1];
    if ( v18 )
    {
      v19 = sub_C1BA30(v18, v58 + 32);
      if ( v19 )
        v17 = v19 + 16;
    }
    v20 = a1[8].m128i_i64[1];
    if ( !v20 )
    {
      v61 = a1[8].m128i_i64;
      goto LABEL_95;
    }
    v21 = *(_DWORD *)v17;
    v22 = a1 + 8;
    while ( 1 )
    {
      if ( *(_DWORD *)(v20 + 32) >= v21 )
      {
        if ( *(_DWORD *)(v20 + 32) != v21 || *(_DWORD *)(v20 + 36) >= *(_DWORD *)(v17 + 4) )
        {
          v22 = (__m128i *)v20;
          v20 = *(_QWORD *)(v20 + 16);
          if ( !v20 )
            goto LABEL_37;
          continue;
        }
        v20 = *(_QWORD *)(v20 + 24);
      }
      else
      {
        v20 = *(_QWORD *)(v20 + 24);
      }
      if ( !v20 )
      {
LABEL_37:
        v61 = v22->m128i_i64;
        if ( &a1[8] == v22
          || v22[2].m128i_i32[0] > v21
          || v22[2].m128i_i32[0] == v21 && *(_DWORD *)(v17 + 4) < v22[2].m128i_i32[1] )
        {
LABEL_95:
          v67[0] = (__int64 *)v17;
          v61 = (_QWORD *)sub_C1D4E0(&a1[7].m128i_i64[1], (__int64)v61, v67);
        }
        v64 = *(const __m128i **)(v58 + 64);
        if ( (const __m128i *)(v58 + 48) == v64 )
          goto LABEL_69;
        v63 = v61 + 6;
        while ( 2 )
        {
          v23 = (_QWORD *)v61[7];
          if ( !v23 )
          {
            v24 = v61 + 6;
            goto LABEL_87;
          }
          v24 = v61 + 6;
          v25 = v64[2].m128i_u64[1];
          v26 = (const void *)v64[2].m128i_i64[0];
          while ( 2 )
          {
            while ( 2 )
            {
              v29 = v23[5];
              v30 = (const void *)v23[4];
              if ( v25 >= v29 )
              {
                if ( v30 != v26 )
                {
                  v27 = v23[5];
                  if ( !v30 )
                    goto LABEL_54;
                  goto LABEL_46;
                }
LABEL_48:
                if ( v25 != v29 )
                {
LABEL_49:
                  if ( v25 > v29 )
                    goto LABEL_54;
                }
LABEL_50:
                v24 = v23;
                v23 = (_QWORD *)v23[2];
                if ( !v23 )
                  goto LABEL_56;
                continue;
              }
              break;
            }
            if ( v30 == v26 )
              goto LABEL_49;
            v27 = v25;
            if ( !v30 )
            {
LABEL_54:
              v23 = (_QWORD *)v23[3];
              goto LABEL_55;
            }
LABEL_46:
            if ( !v26 )
              goto LABEL_50;
            v28 = memcmp(v30, v26, v27);
            if ( !v28 )
              goto LABEL_48;
            if ( v28 >= 0 )
              goto LABEL_50;
            v23 = (_QWORD *)v23[3];
LABEL_55:
            if ( v23 )
              continue;
            break;
          }
LABEL_56:
          if ( v63 == v24 )
            goto LABEL_87;
          v31 = v24[5];
          v32 = (const void *)v24[4];
          if ( v25 > v31 )
          {
            if ( v32 == v26 )
              goto LABEL_64;
            v33 = v24[5];
LABEL_60:
            if ( !v26 )
              goto LABEL_87;
            if ( v32 )
            {
              v56 = v24[5];
              v34 = memcmp(v26, v32, v33);
              v31 = v56;
              if ( !v34 )
                goto LABEL_63;
              if ( v34 < 0 )
                goto LABEL_87;
            }
LABEL_65:
            v35 = v24 + 6;
          }
          else
          {
            if ( v32 != v26 )
            {
              v33 = v25;
              goto LABEL_60;
            }
LABEL_63:
            v35 = v24 + 6;
            if ( v25 != v31 )
            {
LABEL_64:
              if ( v25 >= v31 )
                goto LABEL_65;
LABEL_87:
              v42 = v24;
              v43 = sub_22077B0(224);
              v35 = (_QWORD *)(v43 + 48);
              *(__m128i *)(v43 + 32) = _mm_loadu_si128(v64 + 2);
              memset((void *)(v43 + 48), 0, 0xB0u);
              *(_QWORD *)(v43 + 144) = v43 + 128;
              *(_QWORD *)(v43 + 152) = v43 + 128;
              *(_QWORD *)(v43 + 192) = v43 + 176;
              *(_QWORD *)(v43 + 200) = v43 + 176;
              v44 = sub_C1C960(v61 + 5, v42, (const void **)(v43 + 32));
              v46 = v45;
              if ( v45 )
              {
                if ( v63 == v45 || v44 )
                {
LABEL_90:
                  v47 = 1;
                  goto LABEL_91;
                }
                v48 = v45[5];
                v49 = *(_QWORD *)(v43 + 40);
                v50 = (const void *)v45[4];
                v51 = *(const void **)(v43 + 32);
                if ( v48 < v49 )
                {
                  if ( v50 == v51 )
                    goto LABEL_108;
                  v52 = v45[5];
LABEL_104:
                  if ( !v51 )
                    goto LABEL_90;
                  if ( v50 )
                  {
                    v57 = v46;
                    v53 = memcmp(v51, v50, v52);
                    v46 = v57;
                    v47 = v53 >> 31;
                    if ( !v53 )
                      goto LABEL_107;
                  }
                  else
                  {
                    v47 = 0;
                  }
                }
                else
                {
                  if ( v50 != v51 )
                  {
                    v52 = *(_QWORD *)(v43 + 40);
                    goto LABEL_104;
                  }
LABEL_107:
                  v47 = 0;
                  if ( v48 != v49 )
LABEL_108:
                    v47 = v48 > v49;
                }
LABEL_91:
                sub_220F040(v47, v43, v46, v63);
                ++v61[10];
              }
              else
              {
                v35 = v44 + 6;
                sub_C1A550(0);
                j_j___libc_free_0(v43, 224);
              }
            }
          }
          v36 = sub_C1D5C0(v35, &v64[3], a3);
          if ( v62 )
            v36 = v62;
          v62 = v36;
          v64 = (const __m128i *)sub_220EF30(v64);
          if ( (const __m128i *)(v58 + 48) != v64 )
            continue;
          break;
        }
LABEL_69:
        v58 = sub_220EF30(v58);
        if ( v55 == (const __m128i *)v58 )
          return v62;
        goto LABEL_27;
      }
    }
  }
  return v62;
}
