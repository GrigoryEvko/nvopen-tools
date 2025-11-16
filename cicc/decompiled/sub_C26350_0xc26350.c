// Function: sub_C26350
// Address: 0xc26350
//
__int64 __fastcall sub_C26350(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *i; // rdx
  __int64 v6; // rax
  char v7; // al
  unsigned __int64 v8; // rdx
  char v9; // r15
  __int64 v10; // rcx
  unsigned int v11; // eax
  unsigned __int64 v12; // rsi
  __int64 v13; // rsi
  __int64 v14; // r12
  __m128i *v15; // rsi
  __int64 result; // rax
  unsigned int v17; // esi
  __int64 v18; // r8
  __int64 v19; // rdx
  __int64 v20; // r10
  unsigned int v21; // r9d
  _QWORD *v22; // rax
  __int64 v23; // rdi
  __int64 *v24; // rax
  unsigned int v25; // ecx
  unsigned int v26; // eax
  _QWORD *v27; // rdi
  int v28; // r12d
  _QWORD *v29; // rax
  const __m128i *v30; // rsi
  const __m128i *v31; // rax
  const __m128i *v32; // rdi
  __m128i *v33; // r13
  __int64 v34; // r14
  signed __int64 v35; // r12
  __int64 v36; // rax
  __m128i *v37; // rdx
  _QWORD *v38; // rcx
  int v39; // eax
  int v40; // eax
  int v41; // esi
  int v42; // esi
  __int64 v43; // r10
  unsigned int v44; // edi
  __int64 v45; // r9
  _QWORD *v46; // r11
  int v47; // esi
  int v48; // esi
  __int64 v49; // r10
  unsigned int v50; // edi
  __int64 v51; // r9
  unsigned __int64 v52; // rax
  unsigned __int64 v53; // rdi
  _QWORD *v54; // rax
  __int64 v55; // rdx
  _QWORD *j; // rdx
  __int64 v57; // [rsp+0h] [rbp-D0h]
  __int64 v58; // [rsp+0h] [rbp-D0h]
  int v59; // [rsp+8h] [rbp-C8h]
  __int64 v60; // [rsp+8h] [rbp-C8h]
  int v61; // [rsp+8h] [rbp-C8h]
  __int64 v62; // [rsp+8h] [rbp-C8h]
  int v63; // [rsp+8h] [rbp-C8h]
  __int64 v64; // [rsp+10h] [rbp-C0h]
  unsigned __int64 v65; // [rsp+20h] [rbp-B0h] BYREF
  char v66; // [rsp+30h] [rbp-A0h]
  __int64 v67; // [rsp+40h] [rbp-90h] BYREF
  char v68; // [rsp+50h] [rbp-80h]
  __m128i v69; // [rsp+60h] [rbp-70h] BYREF
  __m128i v70; // [rsp+70h] [rbp-60h] BYREF
  __int64 v71; // [rsp+80h] [rbp-50h]
  __int64 v72; // [rsp+88h] [rbp-48h]
  char v73; // [rsp+90h] [rbp-40h]

  v2 = *(_DWORD *)(a1 + 448);
  ++*(_QWORD *)(a1 + 432);
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 452) )
      goto LABEL_7;
    v3 = *(unsigned int *)(a1 + 456);
    if ( (unsigned int)v3 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 440), 16LL * (unsigned int)v3, 8);
      *(_QWORD *)(a1 + 440) = 0;
      *(_QWORD *)(a1 + 448) = 0;
      *(_DWORD *)(a1 + 456) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v25 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 456);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v25 = 64;
  if ( v25 >= (unsigned int)v3 )
  {
LABEL_4:
    v4 = *(_QWORD **)(a1 + 440);
    for ( i = &v4[2 * v3]; i != v4; v4 += 2 )
      *v4 = -1;
    *(_QWORD *)(a1 + 448) = 0;
    goto LABEL_7;
  }
  v26 = v2 - 1;
  if ( v26 )
  {
    _BitScanReverse(&v26, v26);
    v27 = *(_QWORD **)(a1 + 440);
    v28 = 1 << (33 - (v26 ^ 0x1F));
    if ( v28 < 64 )
      v28 = 64;
    if ( (_DWORD)v3 == v28 )
    {
      *(_QWORD *)(a1 + 448) = 0;
      v29 = &v27[2 * (unsigned int)v3];
      do
      {
        if ( v27 )
          *v27 = -1;
        v27 += 2;
      }
      while ( v29 != v27 );
      goto LABEL_7;
    }
  }
  else
  {
    v27 = *(_QWORD **)(a1 + 440);
    v28 = 64;
  }
  sub_C7D6A0(v27, 16LL * (unsigned int)v3, 8);
  v52 = ((((((((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
           | (4 * v28 / 3u + 1)
           | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 4)
         | (((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
         | (4 * v28 / 3u + 1)
         | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
         | (4 * v28 / 3u + 1)
         | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 4)
       | (((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
       | (4 * v28 / 3u + 1)
       | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 16;
  v53 = (v52
       | (((((((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
           | (4 * v28 / 3u + 1)
           | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 4)
         | (((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
         | (4 * v28 / 3u + 1)
         | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
         | (4 * v28 / 3u + 1)
         | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 4)
       | (((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
       | (4 * v28 / 3u + 1)
       | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 456) = v53;
  v54 = (_QWORD *)sub_C7D670(16 * v53, 8);
  v55 = *(unsigned int *)(a1 + 456);
  *(_QWORD *)(a1 + 448) = 0;
  *(_QWORD *)(a1 + 440) = v54;
  for ( j = &v54[2 * v55]; j != v54; v54 += 2 )
  {
    if ( v54 )
      *v54 = -1;
  }
LABEL_7:
  v6 = *(_QWORD *)(a1 + 464);
  if ( *(_QWORD *)(a1 + 472) != v6 )
    *(_QWORD *)(a1 + 472) = v6;
  sub_C21E40((__int64)&v65, (_QWORD *)a1);
  if ( (v66 & 1) == 0 || (result = (unsigned int)v65, !(_DWORD)v65) )
  {
    v64 = a1 + 432;
    v7 = sub_C20C60(a1);
    v8 = v65;
    v9 = v7;
    if ( v7 )
    {
      if ( v65 > 0x2AAAAAAAAAAAAAALL )
        sub_4262D8((__int64)"vector::reserve");
      v30 = *(const __m128i **)(a1 + 464);
      v31 = v30;
      if ( v65 > 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(a1 + 480) - (_QWORD)v30) >> 4) )
      {
        v32 = *(const __m128i **)(a1 + 472);
        v33 = 0;
        v34 = 3 * v65;
        v35 = (char *)v32 - (char *)v30;
        if ( v65 )
        {
          v36 = sub_22077B0(48 * v65);
          v30 = *(const __m128i **)(a1 + 464);
          v32 = *(const __m128i **)(a1 + 472);
          v33 = (__m128i *)v36;
          v31 = v30;
        }
        if ( v30 != v32 )
        {
          v37 = v33;
          do
          {
            if ( v37 )
            {
              *v37 = _mm_loadu_si128(v31);
              v37[1] = _mm_loadu_si128(v31 + 1);
              v37[2] = _mm_loadu_si128(v31 + 2);
            }
            v31 += 3;
            v37 += 3;
          }
          while ( v31 != v32 );
          v32 = v30;
        }
        if ( v32 )
          j_j___libc_free_0(v32, *(_QWORD *)(a1 + 480) - (_QWORD)v32);
        *(_QWORD *)(a1 + 464) = v33;
        v8 = v65;
        *(_QWORD *)(a1 + 472) = (char *)v33 + v35;
        *(_QWORD *)(a1 + 480) = &v33[v34];
      }
    }
    else
    {
      v10 = *(_QWORD *)(a1 + 432) + 1LL;
      if ( (_DWORD)v65 )
      {
        v11 = 4 * v65;
        *(_QWORD *)(a1 + 432) = v10;
        v12 = (((((((v11 / 3 + 1) | ((unsigned __int64)(v11 / 3 + 1) >> 1)) >> 2)
                | (v11 / 3 + 1)
                | ((unsigned __int64)(v11 / 3 + 1) >> 1)) >> 4)
              | (((v11 / 3 + 1) | ((unsigned __int64)(v11 / 3 + 1) >> 1)) >> 2)
              | (v11 / 3 + 1)
              | ((unsigned __int64)(v11 / 3 + 1) >> 1)) >> 8)
            | (((((v11 / 3 + 1) | ((unsigned __int64)(v11 / 3 + 1) >> 1)) >> 2)
              | (v11 / 3 + 1)
              | ((unsigned __int64)(v11 / 3 + 1) >> 1)) >> 4)
            | (((v11 / 3 + 1) | ((unsigned __int64)(v11 / 3 + 1) >> 1)) >> 2)
            | (v11 / 3 + 1)
            | ((unsigned __int64)(v11 / 3 + 1) >> 1);
        v13 = ((v12 >> 16) | v12) + 1;
        if ( *(_DWORD *)(a1 + 456) >= (unsigned int)v13 )
        {
LABEL_15:
          v14 = 0;
          while ( 1 )
          {
            while ( 1 )
            {
              sub_C22680((__int64)&v69, a1);
              if ( (v73 & 1) != 0 )
              {
                result = v69.m128i_u32[0];
                if ( v69.m128i_i32[0] )
                  return result;
              }
              sub_C21E40((__int64)&v67, (_QWORD *)a1);
              if ( (v68 & 1) != 0 )
              {
                result = (unsigned int)v67;
                if ( (_DWORD)v67 )
                  return result;
              }
              if ( !v9 )
                break;
              v15 = *(__m128i **)(a1 + 472);
              if ( v15 == *(__m128i **)(a1 + 480) )
              {
                sub_C22890((const __m128i **)(a1 + 464), v15, &v69, &v67);
              }
              else
              {
                if ( v15 )
                {
                  *v15 = _mm_loadu_si128(&v69);
                  v15[1] = _mm_loadu_si128(&v70);
                  v15[2].m128i_i64[0] = v71;
                  v15[2].m128i_i64[1] = v67;
                  v15 = *(__m128i **)(a1 + 472);
                }
                *(_QWORD *)(a1 + 472) = v15 + 3;
              }
              if ( v65 <= ++v14 )
                goto LABEL_30;
            }
            v17 = *(_DWORD *)(a1 + 456);
            v18 = v67;
            v19 = v72;
            if ( !v17 )
              break;
            v20 = *(_QWORD *)(a1 + 440);
            v21 = v72 & (v17 - 1);
            v22 = (_QWORD *)(v20 + 16LL * v21);
            v23 = *v22;
            if ( v72 != *v22 )
            {
              v59 = 1;
              v38 = 0;
              while ( v23 != -1 )
              {
                if ( !v38 && v23 == -2 )
                  v38 = v22;
                v21 = (v17 - 1) & (v59 + v21);
                v22 = (_QWORD *)(v20 + 16LL * v21);
                v23 = *v22;
                if ( v72 == *v22 )
                  goto LABEL_28;
                ++v59;
              }
              if ( !v38 )
                v38 = v22;
              v39 = *(_DWORD *)(a1 + 448);
              ++*(_QWORD *)(a1 + 432);
              v40 = v39 + 1;
              if ( 4 * v40 < 3 * v17 )
              {
                if ( v17 - *(_DWORD *)(a1 + 452) - v40 <= v17 >> 3 )
                {
                  v58 = v19;
                  v62 = v18;
                  sub_C26170(v64, v17);
                  v47 = *(_DWORD *)(a1 + 456);
                  if ( !v47 )
                  {
LABEL_106:
                    ++*(_DWORD *)(a1 + 448);
                    BUG();
                  }
                  v19 = v58;
                  v48 = v47 - 1;
                  v49 = *(_QWORD *)(a1 + 440);
                  v18 = v62;
                  v50 = v58 & v48;
                  v40 = *(_DWORD *)(a1 + 448) + 1;
                  v38 = (_QWORD *)(v49 + 16LL * ((unsigned int)v58 & v48));
                  v51 = *v38;
                  if ( v58 != *v38 )
                  {
                    v63 = 1;
                    v46 = 0;
                    while ( v51 != -1 )
                    {
                      if ( !v46 && v51 == -2 )
                        v46 = v38;
                      v50 = v48 & (v63 + v50);
                      v38 = (_QWORD *)(v49 + 16LL * v50);
                      v51 = *v38;
                      if ( v58 == *v38 )
                        goto LABEL_68;
                      ++v63;
                    }
                    goto LABEL_76;
                  }
                }
                goto LABEL_68;
              }
LABEL_72:
              v57 = v19;
              v60 = v18;
              sub_C26170(v64, 2 * v17);
              v41 = *(_DWORD *)(a1 + 456);
              if ( !v41 )
                goto LABEL_106;
              v19 = v57;
              v42 = v41 - 1;
              v43 = *(_QWORD *)(a1 + 440);
              v18 = v60;
              v44 = v57 & v42;
              v40 = *(_DWORD *)(a1 + 448) + 1;
              v38 = (_QWORD *)(v43 + 16LL * ((unsigned int)v57 & v42));
              v45 = *v38;
              if ( v57 != *v38 )
              {
                v61 = 1;
                v46 = 0;
                while ( v45 != -1 )
                {
                  if ( !v46 && v45 == -2 )
                    v46 = v38;
                  v44 = v42 & (v61 + v44);
                  v38 = (_QWORD *)(v43 + 16LL * v44);
                  v45 = *v38;
                  if ( v57 == *v38 )
                    goto LABEL_68;
                  ++v61;
                }
LABEL_76:
                if ( v46 )
                  v38 = v46;
              }
LABEL_68:
              *(_DWORD *)(a1 + 448) = v40;
              if ( *v38 != -1 )
                --*(_DWORD *)(a1 + 452);
              *v38 = v19;
              v24 = v38 + 1;
              v38[1] = 0;
              goto LABEL_29;
            }
LABEL_28:
            v24 = v22 + 1;
LABEL_29:
            *v24 = v18;
            if ( v65 <= ++v14 )
              goto LABEL_30;
          }
          ++*(_QWORD *)(a1 + 432);
          goto LABEL_72;
        }
        sub_C26170(v64, v13);
        v8 = v65;
      }
      else
      {
        *(_QWORD *)(a1 + 432) = v10;
      }
    }
    if ( !v8 )
    {
LABEL_30:
      sub_C1AFD0();
      return 0;
    }
    goto LABEL_15;
  }
  return result;
}
