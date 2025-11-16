// Function: sub_22A5210
// Address: 0x22a5210
//
__int64 __fastcall sub_22A5210(__int64 a1, __int64 a2, unsigned __int64 *a3)
{
  unsigned __int64 v5; // rdx
  __m128i *v6; // rax
  __int64 v7; // r14
  __int64 v8; // rbx
  unsigned int v9; // esi
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rcx
  __int64 v13; // r11
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdi
  char *v17; // rax
  __int64 **v18; // r12
  __int64 **v19; // r13
  __int64 *v20; // r15
  __int64 v21; // rbx
  char *v22; // rax
  char *v23; // rcx
  __m128i *v25; // rsi
  char v26; // dl
  unsigned __int64 v27; // rax
  __int64 v28; // r15
  unsigned int i; // r14d
  __int64 v30; // rcx
  unsigned int v31; // eax
  __int64 v32; // rax
  __int64 *v33; // rbx
  __int64 *v34; // r14
  unsigned int v35; // esi
  __int64 v36; // rdi
  __int64 v37; // r8
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rcx
  __int64 v41; // r15
  __int64 v42; // rax
  __int64 v43; // r13
  __int64 v44; // r14
  __int64 v45; // r15
  __int64 v46; // rdx
  __int64 v47; // rcx
  unsigned int v48; // edx
  __int64 v49; // r12
  unsigned int v50; // edx
  __int64 v51; // rdx
  int v52; // r10d
  int v53; // edx
  int v54; // r12d
  int v55; // r12d
  __int64 v56; // r10
  unsigned int v57; // esi
  __int64 v58; // rdi
  int v59; // r11d
  int v60; // r11d
  __int64 v61; // rsi
  unsigned int v62; // r12d
  int v63; // edi
  unsigned __int64 v64; // rdx
  __m128i *v65; // rax
  __int64 v66; // r9
  __m128i *v67; // rdi
  int v68; // r14d
  unsigned __int32 v69; // edx
  __int64 v70; // r15
  __int64 *v71; // r13
  __int64 v72; // rbx
  __int64 v73; // rdx
  __int64 v74; // r10
  __m128i *v75; // rdi
  int v76; // r12d
  __m128i *v77; // rsi
  __int64 *v78; // rcx
  int v79; // r11d
  __int64 v80; // r10
  int v81; // edx
  int v82; // r10d
  int v83; // r10d
  __int64 v84; // rcx
  unsigned int v85; // r12d
  int v86; // esi
  __int64 v87; // rdi
  int v88; // r11d
  int v89; // r11d
  __int64 v90; // r9
  unsigned int v91; // ecx
  int v92; // edi
  __int64 v93; // rsi
  __int64 v94; // rdi
  int v95; // r10d
  __int64 v96; // rdi
  int v97; // r9d
  __int64 v98; // rsi
  __int64 v99; // [rsp+10h] [rbp-3C0h]
  __int64 v100; // [rsp+18h] [rbp-3B8h]
  __int64 v101; // [rsp+18h] [rbp-3B8h]
  __int64 v102; // [rsp+20h] [rbp-3B0h]
  __m128i *v103; // [rsp+20h] [rbp-3B0h]
  __int64 v104; // [rsp+28h] [rbp-3A8h]
  __int64 v105; // [rsp+28h] [rbp-3A8h]
  __int64 v107; // [rsp+38h] [rbp-398h]
  __int64 v108; // [rsp+40h] [rbp-390h]
  __int64 v109; // [rsp+40h] [rbp-390h]
  __int64 v110; // [rsp+40h] [rbp-390h]
  char v111; // [rsp+48h] [rbp-388h]
  int v112; // [rsp+48h] [rbp-388h]
  __int64 *v113; // [rsp+48h] [rbp-388h]
  const __m128i *v114; // [rsp+50h] [rbp-380h] BYREF
  __m128i *v115; // [rsp+58h] [rbp-378h]
  const __m128i *v116; // [rsp+60h] [rbp-370h]
  __int64 v117; // [rsp+70h] [rbp-360h] BYREF
  char *v118; // [rsp+78h] [rbp-358h]
  __int64 v119; // [rsp+80h] [rbp-350h]
  int v120; // [rsp+88h] [rbp-348h]
  char v121; // [rsp+8Ch] [rbp-344h]
  char v122; // [rsp+90h] [rbp-340h] BYREF
  __m128i v123; // [rsp+190h] [rbp-240h] BYREF
  __m128i v124; // [rsp+1A0h] [rbp-230h] BYREF

  v5 = *a3;
  v124 = (__m128i)(unsigned __int64)a3;
  v123 = (__m128i)v5;
  v114 = 0;
  v115 = 0;
  v116 = 0;
  v117 = 0;
  v118 = &v122;
  v119 = 32;
  v120 = 0;
  v121 = 1;
  sub_22A4A60((unsigned __int64 *)&v114, 0, &v123);
  v6 = v115;
LABEL_2:
  v7 = v6[-2].m128i_i64[0];
  v8 = v6[-1].m128i_i64[0];
  v102 = v6[-2].m128i_i64[1];
  v99 = v6[-1].m128i_i64[1];
  v9 = *(_DWORD *)(a1 + 24);
  if ( !v9 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_93;
  }
  v10 = *(_QWORD *)(a1 + 8);
  v11 = v9 - 1;
  v12 = 1;
  v13 = 0;
  v14 = (unsigned int)v11 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v15 = v10 + 56 * v14;
  v16 = *(_QWORD *)v15;
  if ( v7 != *(_QWORD *)v15 )
  {
    while ( v16 != -4096 )
    {
      if ( v16 == -8192 && !v13 )
        v13 = v15;
      v52 = v12 + 1;
      v14 = (unsigned int)v11 & ((_DWORD)v14 + (_DWORD)v12);
      v12 = (unsigned int)v14;
      v15 = v10 + 56LL * (unsigned int)v14;
      v16 = *(_QWORD *)v15;
      if ( v7 == *(_QWORD *)v15 )
        goto LABEL_4;
      LODWORD(v12) = v52;
    }
    if ( v13 )
      v15 = v13;
    ++*(_QWORD *)a1;
    v53 = *(_DWORD *)(a1 + 16) + 1;
    if ( 4 * v53 < 3 * v9 )
    {
      v12 = a1;
      v10 = v9 >> 3;
      if ( v9 - *(_DWORD *)(a1 + 20) - v53 <= (unsigned int)v10 )
      {
        sub_22A4BF0(a1, v9);
        v59 = *(_DWORD *)(a1 + 24);
        if ( !v59 )
          goto LABEL_164;
        v60 = v59 - 1;
        v11 = *(_QWORD *)(a1 + 8);
        v61 = 0;
        v62 = v60 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v53 = *(_DWORD *)(a1 + 16) + 1;
        v63 = 1;
        v15 = v11 + 56LL * v62;
        v10 = *(_QWORD *)v15;
        if ( v7 != *(_QWORD *)v15 )
        {
          while ( v10 != -4096 )
          {
            if ( v10 == -8192 && !v61 )
              v61 = v15;
            v12 = (unsigned int)(v63 + 1);
            v94 = v60 & (v62 + v63);
            v62 = v94;
            v15 = v11 + 56 * v94;
            v10 = *(_QWORD *)v15;
            if ( v7 == *(_QWORD *)v15 )
              goto LABEL_89;
            v63 = v12;
          }
          if ( v61 )
            v15 = v61;
        }
      }
      goto LABEL_89;
    }
LABEL_93:
    sub_22A4BF0(a1, 2 * v9);
    v54 = *(_DWORD *)(a1 + 24);
    if ( !v54 )
      goto LABEL_164;
    v55 = v54 - 1;
    v56 = *(_QWORD *)(a1 + 8);
    v57 = v55 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v53 = *(_DWORD *)(a1 + 16) + 1;
    v15 = v56 + 56LL * v57;
    v11 = *(_QWORD *)v15;
    if ( v7 != *(_QWORD *)v15 )
    {
      v10 = 1;
      v58 = 0;
      while ( v11 != -4096 )
      {
        if ( !v58 && v11 == -8192 )
          v58 = v15;
        v12 = (unsigned int)(v10 + 1);
        v57 = v55 & (v57 + v10);
        v10 = v57;
        v15 = v56 + 56LL * v57;
        v11 = *(_QWORD *)v15;
        if ( v7 == *(_QWORD *)v15 )
          goto LABEL_89;
        v10 = (unsigned int)v12;
      }
      if ( v58 )
        v15 = v58;
    }
LABEL_89:
    *(_DWORD *)(a1 + 16) = v53;
    if ( *(_QWORD *)v15 != -4096 )
      --*(_DWORD *)(a1 + 20);
    v14 = v15 + 56;
    *(_QWORD *)v15 = v7;
    v104 = v15 + 8;
    *(_QWORD *)(v15 + 40) = v15 + 56;
    *(_QWORD *)(v15 + 48) = 0;
    *(_OWORD *)(v15 + 8) = 0;
    *(_OWORD *)(v15 + 24) = 0;
    goto LABEL_5;
  }
LABEL_4:
  v104 = v15 + 8;
LABEL_5:
  if ( v121 )
  {
    v17 = v118;
    v14 = (__int64)&v118[8 * HIDWORD(v119)];
    if ( v118 != (char *)v14 )
    {
      while ( v7 != *(_QWORD *)v17 )
      {
        v17 += 8;
        if ( (char *)v14 == v17 )
          goto LABEL_47;
      }
      goto LABEL_10;
    }
LABEL_47:
    if ( HIDWORD(v119) < (unsigned int)v119 )
    {
      ++HIDWORD(v119);
      *(_QWORD *)v14 = v7;
      ++v117;
LABEL_34:
      v27 = *(_QWORD *)(v7 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v27 != v7 + 48 )
      {
        if ( !v27 )
          BUG();
        v28 = v27 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v27 - 24) - 30 <= 0xA )
        {
          v112 = sub_B46E30(v28);
          if ( v112 )
          {
            v109 = v7;
            for ( i = 0; i != v112; ++i )
            {
              v32 = sub_B46EC0(v28, i);
              v123.m128i_i64[0] = v32;
              if ( v32 )
              {
                v30 = (unsigned int)(*(_DWORD *)(v32 + 44) + 1);
                v31 = *(_DWORD *)(v32 + 44) + 1;
              }
              else
              {
                v30 = 0;
                v31 = 0;
              }
              if ( *(_DWORD *)(a2 + 32) <= v31 )
                BUG();
              if ( v8 != *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 24) + 8 * v30) + 8LL) )
                sub_22A4F80(v104, v123.m128i_i64);
            }
            v7 = v109;
          }
        }
      }
      goto LABEL_10;
    }
  }
  sub_C8CC70((__int64)&v117, v7, v14, v12, v10, v11);
  if ( v26 )
    goto LABEL_34;
LABEL_10:
  v18 = *(__int64 ***)(v8 + 24);
  v111 = 0;
  if ( &v18[*(unsigned int *)(v8 + 32)] == v18 )
    goto LABEL_49;
  v108 = v8;
  v100 = a2;
  v19 = &v18[*(unsigned int *)(v8 + 32)];
  do
  {
    while ( 1 )
    {
      v20 = *v18;
      v21 = **v18;
      if ( v121 )
        break;
      if ( !sub_C8CA60((__int64)&v117, v21) )
        goto LABEL_27;
LABEL_17:
      if ( ++v18 == v19 )
        goto LABEL_18;
    }
    v22 = v118;
    v23 = &v118[8 * HIDWORD(v119)];
    if ( v118 != v23 )
    {
      while ( v21 != *(_QWORD *)v22 )
      {
        v22 += 8;
        if ( v23 == v22 )
          goto LABEL_27;
      }
      goto LABEL_17;
    }
LABEL_27:
    v123.m128i_i64[0] = v21;
    v123.m128i_i64[1] = v7;
    v25 = v115;
    v124.m128i_i64[0] = (__int64)v20;
    v124.m128i_i64[1] = v108;
    if ( v115 == v116 )
    {
      sub_22A4A60((unsigned __int64 *)&v114, v115, &v123);
    }
    else
    {
      if ( v115 )
      {
        *v115 = _mm_load_si128(&v123);
        v25[1] = _mm_load_si128(&v124);
        v25 = v115;
      }
      v115 = v25 + 2;
    }
    ++v18;
    v111 = 1;
  }
  while ( v18 != v19 );
LABEL_18:
  a2 = v100;
  if ( v111 )
    goto LABEL_19;
LABEL_49:
  if ( v102 )
  {
    v33 = *(__int64 **)(v104 + 32);
    v34 = &v33[*(unsigned int *)(v104 + 40)];
    v35 = *(_DWORD *)(a1 + 24);
    if ( v35 )
    {
      v36 = *(_QWORD *)(a1 + 8);
      v37 = v35 - 1;
      LODWORD(v38) = v37 & (((unsigned int)v102 >> 9) ^ ((unsigned int)v102 >> 4));
      v39 = v36 + 56LL * (unsigned int)v38;
      v40 = *(_QWORD *)v39;
      if ( v102 == *(_QWORD *)v39 )
      {
LABEL_52:
        v41 = v39 + 8;
        goto LABEL_53;
      }
      v79 = 1;
      v80 = 0;
      while ( v40 != -4096 )
      {
        if ( !v80 && v40 == -8192 )
          v80 = v39;
        v38 = (unsigned int)v37 & ((_DWORD)v38 + v79);
        v39 = v36 + 56 * v38;
        v40 = *(_QWORD *)v39;
        if ( v102 == *(_QWORD *)v39 )
          goto LABEL_52;
        ++v79;
      }
      if ( v80 )
        v39 = v80;
      ++*(_QWORD *)a1;
      v81 = *(_DWORD *)(a1 + 16) + 1;
      if ( 4 * v81 < 3 * v35 )
      {
        if ( v35 - *(_DWORD *)(a1 + 20) - v81 > v35 >> 3 )
        {
LABEL_121:
          *(_DWORD *)(a1 + 16) = v81;
          if ( *(_QWORD *)v39 != -4096 )
            --*(_DWORD *)(a1 + 20);
          *(_QWORD *)(v39 + 48) = 0;
          *(_QWORD *)(v39 + 40) = v39 + 56;
          v41 = v39 + 8;
          *(_QWORD *)v39 = v102;
          *(_OWORD *)(v39 + 8) = 0;
          *(_OWORD *)(v39 + 24) = 0;
LABEL_53:
          if ( v34 == v33 )
            goto LABEL_71;
          v42 = a2;
          v113 = v34;
          v43 = v41;
          v44 = v99;
          v45 = v42;
          while ( 2 )
          {
            v46 = *v33;
            if ( *v33 )
            {
              v47 = (unsigned int)(*(_DWORD *)(v46 + 44) + 1);
              v48 = *(_DWORD *)(v46 + 44) + 1;
            }
            else
            {
              v47 = 0;
              v48 = 0;
            }
            if ( *(_DWORD *)(v45 + 32) > v48 )
            {
              v49 = *(_QWORD *)(*(_QWORD *)(v45 + 24) + 8 * v47);
              if ( v44 != v49 && v49 != 0 )
              {
                if ( v44 )
                {
                  if ( v44 == *(_QWORD *)(v49 + 8) )
                    goto LABEL_69;
                  if ( v49 != *(_QWORD *)(v44 + 8) && *(_DWORD *)(v44 + 16) < *(_DWORD *)(v49 + 16) )
                  {
                    if ( *(_BYTE *)(v45 + 112) )
                      goto LABEL_74;
                    v50 = *(_DWORD *)(v45 + 116) + 1;
                    *(_DWORD *)(v45 + 116) = v50;
                    if ( v50 > 0x20 )
                    {
                      v64 = *(_QWORD *)(v45 + 96);
                      v65 = &v124;
                      v123.m128i_i32[3] = 32;
                      v123.m128i_i64[0] = (__int64)&v124;
                      if ( v64 )
                      {
                        v66 = 1;
                        v124 = (__m128i)__PAIR128__(*(_QWORD *)(v64 + 24), v64);
                        v67 = &v124;
                        v123.m128i_i32[2] = 1;
                        v110 = v44;
                        v68 = 1;
                        *(_DWORD *)(v64 + 72) = 0;
                        v69 = 1;
                        v105 = v45;
                        v70 = v43;
                        v71 = v33;
                        v107 = v49;
                        do
                        {
                          v76 = v68++;
                          v77 = &v67[v69 - 1];
                          v78 = (__int64 *)v77->m128i_i64[1];
                          if ( v78 == (__int64 *)(*(_QWORD *)(v77->m128i_i64[0] + 24)
                                                + 8LL * *(unsigned int *)(v77->m128i_i64[0] + 32)) )
                          {
                            --v69;
                            *(_DWORD *)(v77->m128i_i64[0] + 76) = v76;
                            v123.m128i_i32[2] = v69;
                          }
                          else
                          {
                            v72 = *v78;
                            v77->m128i_i64[1] = (__int64)(v78 + 1);
                            v73 = v123.m128i_u32[2];
                            v74 = *(_QWORD *)(v72 + 24);
                            if ( (unsigned __int64)v123.m128i_u32[2] + 1 > v123.m128i_u32[3] )
                            {
                              v101 = *(_QWORD *)(v72 + 24);
                              v103 = v65;
                              sub_C8D5F0((__int64)&v123, v65, v123.m128i_u32[2] + 1LL, 0x10u, v37, v66);
                              v67 = (__m128i *)v123.m128i_i64[0];
                              v73 = v123.m128i_u32[2];
                              v74 = v101;
                              v65 = v103;
                            }
                            v75 = &v67[v73];
                            v75->m128i_i64[0] = v72;
                            v75->m128i_i64[1] = v74;
                            v69 = ++v123.m128i_i32[2];
                            *(_DWORD *)(v72 + 72) = v76;
                            v67 = (__m128i *)v123.m128i_i64[0];
                          }
                        }
                        while ( v69 );
                        v33 = v71;
                        v43 = v70;
                        v45 = v105;
                        v44 = v110;
                        v49 = v107;
                        *(_DWORD *)(v105 + 116) = 0;
                        *(_BYTE *)(v105 + 112) = 1;
                        if ( v67 != v65 )
                          _libc_free((unsigned __int64)v67);
                      }
LABEL_74:
                      if ( *(_DWORD *)(v49 + 72) >= *(_DWORD *)(v44 + 72)
                        && *(_DWORD *)(v49 + 76) <= *(_DWORD *)(v44 + 76) )
                      {
                        goto LABEL_69;
                      }
                    }
                    else
                    {
                      do
                      {
                        v51 = v49;
                        v49 = *(_QWORD *)(v49 + 8);
                      }
                      while ( v49 && *(_DWORD *)(v44 + 16) <= *(_DWORD *)(v49 + 16) );
                      if ( v44 == v51 )
                      {
LABEL_69:
                        if ( ++v33 != v113 )
                          continue;
                        a2 = v45;
LABEL_71:
                        v115 -= 2;
LABEL_19:
                        v6 = v115;
                        if ( v115 == v114 )
                        {
                          v104 = 0;
                          goto LABEL_21;
                        }
                        goto LABEL_2;
                      }
                    }
                  }
                }
              }
            }
            break;
          }
          sub_22A4F80(v43, v33);
          goto LABEL_69;
        }
        sub_22A4BF0(a1, v35);
        v82 = *(_DWORD *)(a1 + 24);
        if ( v82 )
        {
          v83 = v82 - 1;
          v37 = *(_QWORD *)(a1 + 8);
          v84 = 0;
          v85 = v83 & (((unsigned int)v102 >> 9) ^ ((unsigned int)v102 >> 4));
          v81 = *(_DWORD *)(a1 + 16) + 1;
          v86 = 1;
          v39 = v37 + 56LL * v85;
          v87 = *(_QWORD *)v39;
          if ( v102 != *(_QWORD *)v39 )
          {
            while ( v87 != -4096 )
            {
              if ( v87 == -8192 && !v84 )
                v84 = v39;
              v97 = v86 + 1;
              v98 = v83 & (v85 + v86);
              v85 = v98;
              v39 = v37 + 56 * v98;
              v87 = *(_QWORD *)v39;
              if ( v102 == *(_QWORD *)v39 )
                goto LABEL_121;
              v86 = v97;
            }
            if ( v84 )
              v39 = v84;
          }
          goto LABEL_121;
        }
LABEL_164:
        ++*(_DWORD *)(a1 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    sub_22A4BF0(a1, 2 * v35);
    v88 = *(_DWORD *)(a1 + 24);
    if ( v88 )
    {
      v89 = v88 - 1;
      v90 = *(_QWORD *)(a1 + 8);
      v91 = v89 & (((unsigned int)v102 >> 9) ^ ((unsigned int)v102 >> 4));
      v81 = *(_DWORD *)(a1 + 16) + 1;
      v39 = v90 + 56LL * v91;
      v37 = *(_QWORD *)v39;
      if ( v102 != *(_QWORD *)v39 )
      {
        v92 = 1;
        v93 = 0;
        while ( v37 != -4096 )
        {
          if ( v37 == -8192 && !v93 )
            v93 = v39;
          v95 = v92 + 1;
          v96 = v89 & (v91 + v92);
          v91 = v96;
          v39 = v90 + 56 * v96;
          v37 = *(_QWORD *)v39;
          if ( v102 == *(_QWORD *)v39 )
            goto LABEL_121;
          v92 = v95;
        }
        if ( v93 )
          v39 = v93;
      }
      goto LABEL_121;
    }
    goto LABEL_164;
  }
LABEL_21:
  if ( !v121 )
    _libc_free((unsigned __int64)v118);
  if ( v114 )
    j_j___libc_free_0((unsigned __int64)v114);
  return v104;
}
