// Function: sub_1CB7E90
// Address: 0x1cb7e90
//
__int64 __fastcall sub_1CB7E90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // r14
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 v10; // r8
  __int64 i; // r13
  unsigned int v12; // ecx
  __int64 v13; // rax
  const __m128i *v14; // r15
  _QWORD *v15; // rax
  __int64 v16; // rdx
  _BOOL4 v17; // r8d
  __int64 v18; // rax
  _QWORD *v19; // rax
  __int64 v20; // rdx
  _BOOL8 v21; // rdi
  unsigned __int64 v22; // r8
  int v23; // eax
  __int64 v24; // r15
  __int64 v25; // rax
  __int64 v26; // r13
  int v27; // edx
  unsigned __int64 v28; // rsi
  __int64 v29; // r15
  __int64 k; // r15
  _BOOL4 v31; // r13d
  __int64 v32; // rax
  _QWORD *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r15
  __int64 v36; // rdi
  __int64 v37; // r15
  __int64 v38; // r14
  unsigned __int64 v39; // r15
  unsigned __int8 v40; // r9
  const __m128i *v41; // rax
  const __m128i *v42; // rdi
  __int64 v43; // rcx
  __int64 v44; // rdx
  __int64 v45; // r15
  unsigned int v46; // r13d
  int v48; // eax
  int v49; // edx
  __int64 v50; // rbx
  __int64 m; // r15
  unsigned int v52; // r14d
  char v53; // al
  __int64 v54; // r12
  __int64 v55; // rax
  int v56; // eax
  unsigned __int64 v57; // r13
  unsigned int v58; // r13d
  __int64 v59; // rax
  char v60; // dl
  __int64 *v61; // rax
  _QWORD *v62; // r14
  __int64 j; // rbx
  char v64; // al
  __int64 v65; // rax
  unsigned int v66; // edx
  __int64 v67; // rax
  int v68; // eax
  __int64 v69; // rax
  unsigned __int64 v70; // rax
  unsigned int v71; // r13d
  __int64 *v72; // rax
  __int64 *v73; // rax
  __int64 v74; // rax
  __int64 v75; // r13
  __int64 *v76; // rax
  __int64 *v77; // rax
  __int64 *v78; // rax
  __int64 v79; // rax
  __int64 *v80; // rax
  __int64 *v81; // rax
  __int64 v82; // rax
  __int64 v83; // r14
  __int64 *v84; // rax
  __int64 v85; // r8
  __int64 v87; // [rsp+8h] [rbp-88h]
  unsigned __int64 v88; // [rsp+8h] [rbp-88h]
  __int64 v89; // [rsp+10h] [rbp-80h]
  __int64 v90; // [rsp+18h] [rbp-78h]
  __int64 v91; // [rsp+18h] [rbp-78h]
  const __m128i *v92; // [rsp+20h] [rbp-70h]
  __int64 v93; // [rsp+20h] [rbp-70h]
  __int64 v94; // [rsp+28h] [rbp-68h]
  __int64 v95; // [rsp+28h] [rbp-68h]
  __int64 v96; // [rsp+30h] [rbp-60h]
  __int64 v97; // [rsp+30h] [rbp-60h]
  unsigned __int64 v98; // [rsp+30h] [rbp-60h]
  _BOOL4 v99; // [rsp+30h] [rbp-60h]
  __int64 v100; // [rsp+30h] [rbp-60h]
  __int64 v101; // [rsp+30h] [rbp-60h]
  char v102; // [rsp+30h] [rbp-60h]
  unsigned int *v103; // [rsp+30h] [rbp-60h]
  __int64 v104; // [rsp+30h] [rbp-60h]
  __int64 v105; // [rsp+38h] [rbp-58h]
  unsigned int v106; // [rsp+38h] [rbp-58h]
  unsigned int v107; // [rsp+38h] [rbp-58h]
  int v108; // [rsp+4Ch] [rbp-44h] BYREF
  unsigned __int64 v109; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v110; // [rsp+58h] [rbp-38h]

  v3 = a1;
  v4 = a1 + 64;
  v5 = a1 + 56;
  v89 = a2;
  v90 = a1 + 8;
  sub_1CB6AF0(*(_QWORD *)(a1 + 24));
  *(_QWORD *)(a1 + 24) = 0;
  v6 = *(_QWORD *)(a1 + 216);
  *(_QWORD *)(v3 + 32) = v3 + 16;
  *(_QWORD *)(v3 + 40) = v3 + 16;
  *(_QWORD *)(v3 + 48) = 0;
  v94 = v3 + 16;
  sub_1CB6920(v6);
  v7 = *(_QWORD *)(v3 + 120);
  *(_QWORD *)(v3 + 216) = 0;
  *(_QWORD *)(v3 + 224) = v3 + 208;
  *(_QWORD *)(v3 + 232) = v3 + 208;
  *(_QWORD *)(v3 + 240) = 0;
  sub_1CB6CC0(v7);
  v8 = *(_QWORD *)(v3 + 72);
  *(_QWORD *)(v3 + 120) = 0;
  *(_QWORD *)(v3 + 128) = v3 + 112;
  *(_QWORD *)(v3 + 136) = v3 + 112;
  *(_QWORD *)(v3 + 144) = 0;
  v105 = v3 + 112;
  sub_1CB6CC0(v8);
  *(_QWORD *)(v3 + 80) = v4;
  v9 = *(_QWORD *)(v3 + 168);
  *(_QWORD *)(v3 + 72) = 0;
  *(_QWORD *)(v3 + 88) = v4;
  *(_QWORD *)(v3 + 96) = 0;
  v96 = *(_QWORD *)(a2 + 40);
  sub_1CB6AF0(v9);
  *(_QWORD *)(v3 + 168) = 0;
  *(_QWORD *)(v3 + 176) = v3 + 160;
  *(_QWORD *)(v3 + 184) = v3 + 160;
  v92 = (const __m128i *)(v3 + 160);
  *(_QWORD *)(v3 + 192) = 0;
  v10 = *(_QWORD *)(v96 + 16);
  v97 = v96 + 8;
  for ( i = v10; i != v97; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    v12 = *(_DWORD *)(i - 24);
    a2 = (__int64)&v109;
    v109 = i - 56;
    v110 = (unsigned int)(1 << (v12 >> 15)) >> 1;
    sub_1CB74B0(v3 + 152, &v109);
  }
  v13 = sub_1632FA0(*(_QWORD *)(v89 + 40));
  v14 = *(const __m128i **)(v3 + 176);
  *(_QWORD *)(v3 + 280) = v13;
  for ( *(_QWORD *)(v3 + 288) = a3; v14 != v92; v14 = (const __m128i *)sub_220EEE0(v14) )
  {
    v22 = v14[2].m128i_u64[0];
    v23 = *(_DWORD *)v3;
    v109 = v22;
    if ( v14[2].m128i_i32[2] == v23 )
    {
      a2 = v22;
      sub_1CB7560((_QWORD *)v3, v22, *(_DWORD *)(v3 + 4));
    }
    else
    {
      v98 = v22;
      v15 = sub_1819210(v5, &v109);
      if ( v16 )
      {
        v17 = v15 || v4 == v16 || v98 < *(_QWORD *)(v16 + 32);
        v87 = v16;
        v99 = v17;
        v18 = sub_22077B0(40);
        *(_QWORD *)(v18 + 32) = v109;
        sub_220F040(v99, v18, v87, v4);
        ++*(_QWORD *)(v3 + 96);
      }
      v100 = sub_22077B0(48);
      *(__m128i *)(v100 + 32) = _mm_loadu_si128(v14 + 2);
      v19 = sub_1C70290(v90, (unsigned __int64 *)(v100 + 32));
      if ( v20 )
      {
        v21 = v19 || v94 == v20 || *(_QWORD *)(v100 + 32) < *(_QWORD *)(v20 + 32);
        a2 = v100;
        sub_220F040(v21, v100, v20, v94);
        ++*(_QWORD *)(v3 + 48);
      }
      else
      {
        a2 = 48;
        j_j___libc_free_0(v100, 48);
      }
    }
  }
  if ( (*(_BYTE *)(v89 + 18) & 1) != 0 )
  {
    sub_15E08E0(v89, a2);
    v24 = *(_QWORD *)(v89 + 88);
    if ( (*(_BYTE *)(v89 + 18) & 1) != 0 )
      sub_15E08E0(v89, a2);
    v25 = *(_QWORD *)(v89 + 88);
  }
  else
  {
    v24 = *(_QWORD *)(v89 + 88);
    v25 = v24;
  }
  v26 = v25 + 40LL * *(_QWORD *)(v89 + 96);
  while ( v24 != v26 )
  {
    while ( *(_BYTE *)(*(_QWORD *)v24 + 8LL) != 15 )
    {
      v24 += 40;
      if ( v24 == v26 )
        goto LABEL_29;
    }
    v27 = sub_15E0370(v24);
    if ( !v27 )
      v27 = *(_DWORD *)(v3 + 4);
    v28 = v24;
    v24 += 40;
    sub_1CB7560((_QWORD *)v3, v28, v27);
  }
LABEL_29:
  v29 = *(_QWORD *)(v89 + 80);
  v91 = v89 + 72;
  if ( v89 + 72 == v29 )
    goto LABEL_35;
  if ( !v29 )
    BUG();
  while ( *(_QWORD *)(v29 + 24) == v29 + 16 )
  {
    v29 = *(_QWORD *)(v29 + 8);
    if ( v91 == v29 )
      goto LABEL_35;
    if ( !v29 )
      BUG();
  }
  v104 = v5;
  v62 = (_QWORD *)v3;
  j = *(_QWORD *)(v29 + 24);
  while ( v29 != v91 )
  {
    if ( !j )
      BUG();
    if ( *(_BYTE *)(*(_QWORD *)(j - 24) + 8LL) == 15 )
    {
      v66 = *(_DWORD *)v62;
      if ( *(_BYTE *)(j - 8) == 53 )
        v66 = (unsigned int)(1 << *(_WORD *)(j - 6)) >> 1;
      sub_1CB7560(v62, j - 24, v66);
      v64 = *(_BYTE *)(j - 8);
      if ( v64 != 78 )
        goto LABEL_126;
    }
    else
    {
      v64 = *(_BYTE *)(j - 8);
      if ( v64 != 78 )
        goto LABEL_126;
    }
    v67 = *(_QWORD *)(j - 48);
    if ( *(_BYTE *)(v67 + 16) || (*(_BYTE *)(v67 + 33) & 0x20) == 0 )
      goto LABEL_128;
    v68 = *(_DWORD *)(v67 + 36);
    if ( v68 == 137 )
    {
      sub_1CB7560(v62, *(_QWORD *)(j - 24LL * (*(_DWORD *)(j - 4) & 0xFFFFFFF) - 24), *(_DWORD *)v62);
      v64 = *(_BYTE *)(j - 8);
    }
    else
    {
      if ( (v68 & 0xFFFFFFFD) != 0x85 )
        goto LABEL_128;
      v69 = *(_DWORD *)(j - 4) & 0xFFFFFFF;
      v88 = *(_QWORD *)(j + 24 * (1 - v69) - 24);
      sub_1CB7560(v62, *(_QWORD *)(j - 24 * v69 - 24), *(_DWORD *)v62);
      sub_1CB7560(v62, v88, *(_DWORD *)v62);
      v64 = *(_BYTE *)(j - 8);
    }
LABEL_126:
    if ( v64 == 54 || v64 == 55 )
      sub_1CB7560(v62, *(_QWORD *)(j - 48), *(_DWORD *)v62);
LABEL_128:
    for ( j = *(_QWORD *)(j + 8); ; j = *(_QWORD *)(v29 + 24) )
    {
      v65 = v29 - 24;
      if ( !v29 )
        v65 = 0;
      if ( j != v65 + 40 )
        break;
      v29 = *(_QWORD *)(v29 + 8);
      if ( v91 == v29 )
        break;
      if ( !v29 )
        BUG();
    }
  }
  v3 = (__int64)v62;
  v5 = v104;
LABEL_35:
  for ( k = *(_QWORD *)(v3 + 128); k != v105; k = sub_220EF30(k) )
  {
    v33 = sub_1819210(v5, (unsigned __int64 *)(k + 32));
    if ( v34 )
    {
      v31 = v33 || v4 == v34 || *(_QWORD *)(k + 32) < *(_QWORD *)(v34 + 32);
      v101 = v34;
      v32 = sub_22077B0(40);
      *(_QWORD *)(v32 + 32) = *(_QWORD *)(k + 32);
      sub_220F040(v31, v32, v101, v4);
      ++*(_QWORD *)(v3 + 96);
    }
  }
  v35 = *(_QWORD *)(v3 + 120);
  while ( v35 )
  {
    sub_1CB6CC0(*(_QWORD *)(v35 + 24));
    v36 = v35;
    v35 = *(_QWORD *)(v35 + 16);
    j_j___libc_free_0(v36, 40);
  }
  v37 = *(_QWORD *)(v3 + 80);
  *(_QWORD *)(v3 + 120) = 0;
  *(_QWORD *)(v3 + 144) = 0;
  *(_QWORD *)(v3 + 128) = v105;
  *(_QWORD *)(v3 + 136) = v105;
  if ( v37 != v4 )
  {
    v102 = 0;
    v95 = v5;
    v38 = v37;
    while ( 1 )
    {
      while ( 1 )
      {
        v39 = *(_QWORD *)(v38 + 32);
        if ( (unsigned int)sub_1CB76C0((unsigned int *)v3, v39) != *(_DWORD *)(v3 + 4) )
          break;
LABEL_60:
        v38 = sub_220EF30(v38);
        if ( v38 == v4 )
          goto LABEL_61;
      }
      v40 = *(_BYTE *)(v39 + 16);
      if ( v40 <= 0x17u )
      {
        if ( v40 == 17 )
          goto LABEL_60;
        v41 = *(const __m128i **)(v3 + 168);
        if ( v41 )
        {
LABEL_52:
          v42 = v92;
          do
          {
            while ( 1 )
            {
              v43 = v41[1].m128i_i64[0];
              v44 = v41[1].m128i_i64[1];
              if ( v41[2].m128i_i64[0] >= v39 )
                break;
              v41 = (const __m128i *)v41[1].m128i_i64[1];
              if ( !v44 )
                goto LABEL_56;
            }
            v42 = v41;
            v41 = (const __m128i *)v41[1].m128i_i64[0];
          }
          while ( v43 );
LABEL_56:
          if ( v42 != v92 && v42[2].m128i_i64[0] <= v39 )
            goto LABEL_60;
          goto LABEL_58;
        }
LABEL_73:
        if ( v40 == 5 )
        {
          v102 |= sub_1CB7C40(v3, v39);
          v38 = sub_220EF30(v38);
          if ( v38 == v4 )
            goto LABEL_61;
        }
        else
        {
          v48 = sub_1CB76C0((unsigned int *)v3, v39);
          v49 = *(_DWORD *)(v3 + 4);
          if ( v49 == v48 )
            goto LABEL_60;
          sub_1CB7560((_QWORD *)v3, v39, v49);
          v102 = 1;
          v38 = sub_220EF30(v38);
          if ( v38 == v4 )
            goto LABEL_61;
        }
      }
      else
      {
        if ( v40 == 53 )
          goto LABEL_60;
        v41 = *(const __m128i **)(v3 + 168);
        if ( v41 )
          goto LABEL_52;
LABEL_58:
        if ( (unsigned __int8)(v40 - 60) <= 0xCu )
        {
          v102 |= sub_1CB7820((unsigned int *)v3, v39);
          goto LABEL_60;
        }
        switch ( v40 )
        {
          case '8':
            v102 |= sub_1CB7B60(v3, v39);
            v38 = sub_220EF30(v38);
            if ( v38 == v4 )
              goto LABEL_61;
            break;
          case 'M':
            v102 |= sub_1CB78C0((unsigned int *)v3, v39);
            v38 = sub_220EF30(v38);
            if ( v38 == v4 )
              goto LABEL_61;
            break;
          case 'O':
            v102 |= sub_1CB79F0((unsigned int *)v3, v39);
            v38 = sub_220EF30(v38);
            if ( v38 == v4 )
            {
LABEL_61:
              v5 = v95;
              if ( !*(_QWORD *)(v3 + 144) && !v102 )
                goto LABEL_63;
              goto LABEL_35;
            }
            break;
          case 'N':
            v102 |= sub_1CB7A70((unsigned int *)v3, v39);
            v38 = sub_220EF30(v38);
            if ( v38 == v4 )
              goto LABEL_61;
            break;
          default:
            goto LABEL_73;
        }
      }
    }
  }
LABEL_63:
  v45 = *(_QWORD *)(v89 + 80);
  if ( v91 == v45 )
  {
    return 0;
  }
  else
  {
    if ( !v45 )
      BUG();
    while ( *(_QWORD *)(v45 + 24) == v45 + 16 )
    {
      v45 = *(_QWORD *)(v45 + 8);
      if ( v91 == v45 )
        return 0;
      if ( !v45 )
        BUG();
    }
    v46 = 0;
    if ( v91 != v45 )
    {
      v103 = (unsigned int *)v3;
      v50 = v45;
      m = *(_QWORD *)(v45 + 24);
      v52 = 0;
      while ( 1 )
      {
        if ( !m )
          BUG();
        v53 = *(_BYTE *)(m - 8);
        v54 = m - 24;
        switch ( v53 )
        {
          case '7':
            v52 |= sub_1CB7740(v103, m - 24);
            break;
          case '6':
            v52 |= sub_1CB77B0(v103, m - 24);
            break;
          case 'N':
            v55 = *(_QWORD *)(m - 48);
            if ( !*(_BYTE *)(v55 + 16) && (*(_BYTE *)(v55 + 33) & 0x20) != 0 )
            {
              v56 = *(_DWORD *)(v55 + 36);
              if ( (v56 & 0xFFFFFFFD) == 0x85 )
              {
                v57 = *(_QWORD *)(v54 - 24LL * (*(_DWORD *)(m - 4) & 0xFFFFFFF));
                v106 = sub_1CB76C0(v103, *(_QWORD *)(v54 + 24 * (1LL - (*(_DWORD *)(m - 4) & 0xFFFFFFF))));
                v58 = sub_1CB76C0(v103, v57);
                if ( *(_BYTE *)(m - 8) == 78 )
                {
                  v59 = *(_QWORD *)(m - 48);
                  if ( !*(_BYTE *)(v59 + 16) )
                  {
                    v60 = *(_BYTE *)(v59 + 33);
                    if ( (v60 & 0x20) != 0
                      && (*(_DWORD *)(v59 + 36) == 133 || (v60 & 0x20) != 0 && *(_DWORD *)(v59 + 36) == 135) )
                    {
                      if ( v58 > (unsigned int)sub_15603A0((_QWORD *)(m + 32), 0) )
                      {
                        v52 = 1;
                        v109 = *(_QWORD *)(m + 32);
                        v77 = (__int64 *)sub_16498A0(m - 24);
                        *(_QWORD *)(m + 32) = sub_1563C10((__int64 *)&v109, v77, 1, 1);
                        if ( v58 )
                        {
                          v78 = (__int64 *)sub_16498A0(m - 24);
                          v79 = sub_155D330(v78, v58);
                          v108 = 0;
                          v93 = v79;
                          v109 = *(_QWORD *)(m + 32);
                          v80 = (__int64 *)sub_16498A0(m - 24);
                          v109 = sub_1563E10((__int64 *)&v109, v80, &v108, 1, v93);
                          *(_QWORD *)(m + 32) = v109;
                        }
                      }
                      if ( v106 > (unsigned int)sub_15603A0((_QWORD *)(m + 32), 1) )
                      {
                        v109 = *(_QWORD *)(m + 32);
                        v61 = (__int64 *)sub_16498A0(m - 24);
                        *(_QWORD *)(m + 32) = sub_1563C10((__int64 *)&v109, v61, 2, 1);
                        if ( !v106 )
                          goto LABEL_119;
                        v81 = (__int64 *)sub_16498A0(m - 24);
                        v82 = sub_155D330(v81, v106);
                        v108 = 1;
                        v83 = v82;
                        v109 = *(_QWORD *)(m + 32);
                        v84 = (__int64 *)sub_16498A0(m - 24);
                        v85 = v83;
                        v52 = 1;
                        v109 = sub_1563E10((__int64 *)&v109, v84, &v108, 1, v85);
                        *(_QWORD *)(m + 32) = v109;
                      }
                    }
                  }
                }
              }
              else if ( v56 == 137 )
              {
                v107 = sub_15603A0((_QWORD *)(m + 32), 0);
                v70 = sub_1649C60(*(_QWORD *)(v54 - 24LL * (*(_DWORD *)(m - 4) & 0xFFFFFFF)));
                v71 = sub_1CB76C0(v103, v70);
                if ( v71 > v107 )
                {
                  v109 = *(_QWORD *)(m + 32);
                  v72 = (__int64 *)sub_16498A0(m - 24);
                  *(_QWORD *)(m + 32) = sub_1563C10((__int64 *)&v109, v72, 1, 1);
                  if ( v71 )
                  {
                    v73 = (__int64 *)sub_16498A0(m - 24);
                    v74 = sub_155D330(v73, v71);
                    v108 = 0;
                    v75 = v74;
                    v109 = *(_QWORD *)(m + 32);
                    v76 = (__int64 *)sub_16498A0(m - 24);
                    v109 = sub_1563E10((__int64 *)&v109, v76, &v108, 1, v75);
                    *(_QWORD *)(m + 32) = v109;
                  }
LABEL_119:
                  v52 = 1;
                }
              }
            }
            break;
        }
        for ( m = *(_QWORD *)(m + 8); m == v50 - 24 + 40; m = *(_QWORD *)(v50 + 24) )
        {
          v50 = *(_QWORD *)(v50 + 8);
          if ( v91 == v50 )
            return v52;
          if ( !v50 )
            BUG();
        }
        if ( v91 == v50 )
          return v52;
      }
    }
  }
  return v46;
}
