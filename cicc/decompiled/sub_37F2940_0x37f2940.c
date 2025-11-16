// Function: sub_37F2940
// Address: 0x37f2940
//
__int64 __fastcall sub_37F2940(__int64 a1, __int64 a2, unsigned __int64 *a3)
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
  __int64 *v27; // r15
  __int64 *v28; // r12
  __int64 *v29; // r14
  __int64 v30; // rsi
  unsigned int v31; // eax
  __int64 v32; // rax
  __int64 *v33; // rbx
  __int64 *v34; // r15
  unsigned int v35; // esi
  __int64 v36; // rdi
  __int64 v37; // r8
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rcx
  __int64 v41; // r14
  __int64 v42; // r15
  __int64 v43; // r13
  __int64 v44; // r14
  __int64 v45; // rdx
  __int64 v46; // rcx
  unsigned int v47; // edx
  __int64 v48; // r12
  unsigned int v49; // edx
  __int64 v50; // rdx
  int v51; // r10d
  int v52; // edx
  int v53; // r12d
  int v54; // r12d
  __int64 v55; // r10
  unsigned int v56; // esi
  __int64 v57; // rdi
  int v58; // r11d
  int v59; // r11d
  __int64 v60; // rsi
  unsigned int v61; // r12d
  int v62; // edi
  unsigned __int64 v63; // rdx
  __m128i *v64; // rax
  __int64 v65; // r9
  __m128i *v66; // rdi
  int v67; // r14d
  unsigned __int32 v68; // edx
  __int64 v69; // r15
  __int64 *v70; // r13
  __int64 v71; // rbx
  __int64 v72; // rdx
  __int64 v73; // r10
  __m128i *v74; // rdi
  int v75; // r12d
  __m128i *v76; // rsi
  __int64 *v77; // rcx
  int v78; // r11d
  __int64 v79; // r10
  int v80; // edx
  int v81; // r11d
  int v82; // r11d
  __int64 v83; // r9
  unsigned int v84; // ecx
  int v85; // edi
  __int64 v86; // rsi
  int v87; // r10d
  int v88; // r10d
  __int64 v89; // rcx
  unsigned int v90; // r12d
  int v91; // esi
  __int64 v92; // rdi
  __int64 v93; // rdi
  int v94; // r9d
  __int64 v95; // rsi
  int v96; // r10d
  __int64 v97; // rdi
  __int64 v98; // [rsp+8h] [rbp-3C8h]
  __int64 v99; // [rsp+10h] [rbp-3C0h]
  __int64 v100; // [rsp+10h] [rbp-3C0h]
  __int64 v101; // [rsp+18h] [rbp-3B8h]
  __m128i *v102; // [rsp+18h] [rbp-3B8h]
  __int64 v103; // [rsp+28h] [rbp-3A8h]
  __int64 v104; // [rsp+28h] [rbp-3A8h]
  __int64 v106; // [rsp+38h] [rbp-398h]
  __int64 v107; // [rsp+40h] [rbp-390h]
  __int64 v108; // [rsp+40h] [rbp-390h]
  char v109; // [rsp+48h] [rbp-388h]
  __int64 v110; // [rsp+48h] [rbp-388h]
  __int64 *v111; // [rsp+48h] [rbp-388h]
  const __m128i *v112; // [rsp+50h] [rbp-380h] BYREF
  __m128i *v113; // [rsp+58h] [rbp-378h]
  const __m128i *v114; // [rsp+60h] [rbp-370h]
  __int64 v115; // [rsp+70h] [rbp-360h] BYREF
  char *v116; // [rsp+78h] [rbp-358h]
  __int64 v117; // [rsp+80h] [rbp-350h]
  int v118; // [rsp+88h] [rbp-348h]
  char v119; // [rsp+8Ch] [rbp-344h]
  char v120; // [rsp+90h] [rbp-340h] BYREF
  __m128i v121; // [rsp+190h] [rbp-240h] BYREF
  __m128i v122; // [rsp+1A0h] [rbp-230h] BYREF

  v5 = *a3;
  v122 = (__m128i)(unsigned __int64)a3;
  v121 = (__m128i)v5;
  v112 = 0;
  v113 = 0;
  v114 = 0;
  v115 = 0;
  v116 = &v120;
  v117 = 32;
  v118 = 0;
  v119 = 1;
  sub_37F2190((unsigned __int64 *)&v112, 0, &v121);
  v6 = v113;
LABEL_2:
  v7 = v6[-2].m128i_i64[0];
  v8 = v6[-1].m128i_i64[0];
  v103 = v6[-2].m128i_i64[1];
  v98 = v6[-1].m128i_i64[1];
  v9 = *(_DWORD *)(a1 + 24);
  if ( !v9 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_90;
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
      v51 = v12 + 1;
      v14 = (unsigned int)v11 & ((_DWORD)v14 + (_DWORD)v12);
      v12 = (unsigned int)v14;
      v15 = v10 + 56LL * (unsigned int)v14;
      v16 = *(_QWORD *)v15;
      if ( v7 == *(_QWORD *)v15 )
        goto LABEL_4;
      LODWORD(v12) = v51;
    }
    if ( v13 )
      v15 = v13;
    ++*(_QWORD *)a1;
    v52 = *(_DWORD *)(a1 + 16) + 1;
    if ( 4 * v52 < 3 * v9 )
    {
      v12 = a1;
      v10 = v9 >> 3;
      if ( v9 - *(_DWORD *)(a1 + 20) - v52 <= (unsigned int)v10 )
      {
        sub_37F2320(a1, v9);
        v58 = *(_DWORD *)(a1 + 24);
        if ( !v58 )
          goto LABEL_159;
        v59 = v58 - 1;
        v11 = *(_QWORD *)(a1 + 8);
        v60 = 0;
        v61 = v59 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v52 = *(_DWORD *)(a1 + 16) + 1;
        v62 = 1;
        v15 = v11 + 56LL * v61;
        v10 = *(_QWORD *)v15;
        if ( v7 != *(_QWORD *)v15 )
        {
          while ( v10 != -4096 )
          {
            if ( v10 == -8192 && !v60 )
              v60 = v15;
            v12 = (unsigned int)(v62 + 1);
            v93 = v59 & (v61 + v62);
            v61 = v93;
            v15 = v11 + 56 * v93;
            v10 = *(_QWORD *)v15;
            if ( v7 == *(_QWORD *)v15 )
              goto LABEL_86;
            v62 = v12;
          }
          if ( v60 )
            v15 = v60;
        }
      }
      goto LABEL_86;
    }
LABEL_90:
    sub_37F2320(a1, 2 * v9);
    v53 = *(_DWORD *)(a1 + 24);
    if ( !v53 )
      goto LABEL_159;
    v54 = v53 - 1;
    v55 = *(_QWORD *)(a1 + 8);
    v56 = v54 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v52 = *(_DWORD *)(a1 + 16) + 1;
    v15 = v55 + 56LL * v56;
    v11 = *(_QWORD *)v15;
    if ( v7 != *(_QWORD *)v15 )
    {
      v10 = 1;
      v57 = 0;
      while ( v11 != -4096 )
      {
        if ( !v57 && v11 == -8192 )
          v57 = v15;
        v12 = (unsigned int)(v10 + 1);
        v56 = v54 & (v56 + v10);
        v10 = v56;
        v15 = v55 + 56LL * v56;
        v11 = *(_QWORD *)v15;
        if ( v7 == *(_QWORD *)v15 )
          goto LABEL_86;
        v10 = (unsigned int)v12;
      }
      if ( v57 )
        v15 = v57;
    }
LABEL_86:
    *(_DWORD *)(a1 + 16) = v52;
    if ( *(_QWORD *)v15 != -4096 )
      --*(_DWORD *)(a1 + 20);
    v14 = v15 + 56;
    *(_QWORD *)v15 = v7;
    v99 = v15 + 8;
    *(_QWORD *)(v15 + 40) = v15 + 56;
    *(_QWORD *)(v15 + 48) = 0;
    *(_OWORD *)(v15 + 8) = 0;
    *(_OWORD *)(v15 + 24) = 0;
    goto LABEL_5;
  }
LABEL_4:
  v99 = v15 + 8;
LABEL_5:
  if ( v119 )
  {
    v17 = v116;
    v14 = (__int64)&v116[8 * HIDWORD(v117)];
    if ( v116 != (char *)v14 )
    {
      while ( v7 != *(_QWORD *)v17 )
      {
        v17 += 8;
        if ( (char *)v14 == v17 )
          goto LABEL_44;
      }
      goto LABEL_10;
    }
LABEL_44:
    if ( HIDWORD(v117) < (unsigned int)v117 )
    {
      ++HIDWORD(v117);
      *(_QWORD *)v14 = v7;
      ++v115;
LABEL_34:
      v27 = *(__int64 **)(v7 + 112);
      v28 = &v27[*(unsigned int *)(v7 + 120)];
      if ( v28 != v27 )
      {
        v110 = v7;
        v29 = *(__int64 **)(v7 + 112);
        do
        {
          v32 = *v29;
          v121.m128i_i64[0] = v32;
          if ( v32 )
          {
            v30 = (unsigned int)(*(_DWORD *)(v32 + 24) + 1);
            v31 = *(_DWORD *)(v32 + 24) + 1;
          }
          else
          {
            v30 = 0;
            v31 = 0;
          }
          if ( v31 >= *(_DWORD *)(a2 + 32) )
            BUG();
          if ( *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 24) + 8 * v30) + 8LL) != v8 )
            sub_37F26B0(v99, v121.m128i_i64);
          ++v29;
        }
        while ( v28 != v29 );
        v7 = v110;
      }
      goto LABEL_10;
    }
  }
  sub_C8CC70((__int64)&v115, v7, v14, v12, v10, v11);
  if ( v26 )
    goto LABEL_34;
LABEL_10:
  v18 = *(__int64 ***)(v8 + 24);
  v109 = 0;
  if ( &v18[*(unsigned int *)(v8 + 32)] == v18 )
    goto LABEL_46;
  v107 = v8;
  v101 = a2;
  v19 = &v18[*(unsigned int *)(v8 + 32)];
  do
  {
    while ( 1 )
    {
      v20 = *v18;
      v21 = **v18;
      if ( v119 )
        break;
      if ( !sub_C8CA60((__int64)&v115, v21) )
        goto LABEL_27;
LABEL_17:
      if ( ++v18 == v19 )
        goto LABEL_18;
    }
    v22 = v116;
    v23 = &v116[8 * HIDWORD(v117)];
    if ( v116 != v23 )
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
    v121.m128i_i64[0] = v21;
    v121.m128i_i64[1] = v7;
    v25 = v113;
    v122.m128i_i64[0] = (__int64)v20;
    v122.m128i_i64[1] = v107;
    if ( v113 == v114 )
    {
      sub_37F2190((unsigned __int64 *)&v112, v113, &v121);
    }
    else
    {
      if ( v113 )
      {
        *v113 = _mm_load_si128(&v121);
        v25[1] = _mm_load_si128(&v122);
        v25 = v113;
      }
      v113 = v25 + 2;
    }
    ++v18;
    v109 = 1;
  }
  while ( v18 != v19 );
LABEL_18:
  a2 = v101;
  if ( v109 )
    goto LABEL_19;
LABEL_46:
  if ( v103 )
  {
    v33 = *(__int64 **)(v99 + 32);
    v34 = &v33[*(unsigned int *)(v99 + 40)];
    v35 = *(_DWORD *)(a1 + 24);
    if ( v35 )
    {
      v36 = *(_QWORD *)(a1 + 8);
      v37 = v35 - 1;
      LODWORD(v38) = v37 & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
      v39 = v36 + 56LL * (unsigned int)v38;
      v40 = *(_QWORD *)v39;
      if ( v103 == *(_QWORD *)v39 )
      {
LABEL_49:
        v41 = v39 + 8;
        goto LABEL_50;
      }
      v78 = 1;
      v79 = 0;
      while ( v40 != -4096 )
      {
        if ( !v79 && v40 == -8192 )
          v79 = v39;
        v38 = (unsigned int)v37 & ((_DWORD)v38 + v78);
        v39 = v36 + 56 * v38;
        v40 = *(_QWORD *)v39;
        if ( v103 == *(_QWORD *)v39 )
          goto LABEL_49;
        ++v78;
      }
      if ( v79 )
        v39 = v79;
      ++*(_QWORD *)a1;
      v80 = *(_DWORD *)(a1 + 16) + 1;
      if ( 4 * v80 < 3 * v35 )
      {
        if ( v35 - *(_DWORD *)(a1 + 20) - v80 > v35 >> 3 )
        {
LABEL_117:
          *(_DWORD *)(a1 + 16) = v80;
          if ( *(_QWORD *)v39 != -4096 )
            --*(_DWORD *)(a1 + 20);
          *(_QWORD *)(v39 + 48) = 0;
          *(_QWORD *)(v39 + 40) = v39 + 56;
          v41 = v39 + 8;
          *(_QWORD *)v39 = v103;
          *(_OWORD *)(v39 + 8) = 0;
          *(_OWORD *)(v39 + 24) = 0;
LABEL_50:
          if ( v34 == v33 )
            goto LABEL_68;
          v111 = v34;
          v42 = a2;
          v43 = v41;
          v44 = v98;
          while ( 2 )
          {
            v45 = *v33;
            if ( *v33 )
            {
              v46 = (unsigned int)(*(_DWORD *)(v45 + 24) + 1);
              v47 = *(_DWORD *)(v45 + 24) + 1;
            }
            else
            {
              v46 = 0;
              v47 = 0;
            }
            if ( *(_DWORD *)(v42 + 32) > v47 )
            {
              v48 = *(_QWORD *)(*(_QWORD *)(v42 + 24) + 8 * v46);
              if ( v48 != 0 && v44 != v48 )
              {
                if ( v44 )
                {
                  if ( v44 == *(_QWORD *)(v48 + 8) )
                    goto LABEL_66;
                  if ( v48 != *(_QWORD *)(v44 + 8) && *(_DWORD *)(v44 + 16) < *(_DWORD *)(v48 + 16) )
                  {
                    if ( *(_BYTE *)(v42 + 112) )
                      goto LABEL_71;
                    v49 = *(_DWORD *)(v42 + 116) + 1;
                    *(_DWORD *)(v42 + 116) = v49;
                    if ( v49 > 0x20 )
                    {
                      v63 = *(_QWORD *)(v42 + 96);
                      v64 = &v122;
                      v121.m128i_i32[3] = 32;
                      v121.m128i_i64[0] = (__int64)&v122;
                      if ( v63 )
                      {
                        v65 = 1;
                        v122 = (__m128i)__PAIR128__(*(_QWORD *)(v63 + 24), v63);
                        v66 = &v122;
                        v121.m128i_i32[2] = 1;
                        v108 = v44;
                        v67 = 1;
                        *(_DWORD *)(v63 + 72) = 0;
                        v68 = 1;
                        v104 = v42;
                        v69 = v43;
                        v70 = v33;
                        v106 = v48;
                        do
                        {
                          v75 = v67++;
                          v76 = &v66[v68 - 1];
                          v77 = (__int64 *)v76->m128i_i64[1];
                          if ( v77 == (__int64 *)(*(_QWORD *)(v76->m128i_i64[0] + 24)
                                                + 8LL * *(unsigned int *)(v76->m128i_i64[0] + 32)) )
                          {
                            --v68;
                            *(_DWORD *)(v76->m128i_i64[0] + 76) = v75;
                            v121.m128i_i32[2] = v68;
                          }
                          else
                          {
                            v71 = *v77;
                            v76->m128i_i64[1] = (__int64)(v77 + 1);
                            v72 = v121.m128i_u32[2];
                            v73 = *(_QWORD *)(v71 + 24);
                            if ( (unsigned __int64)v121.m128i_u32[2] + 1 > v121.m128i_u32[3] )
                            {
                              v100 = *(_QWORD *)(v71 + 24);
                              v102 = v64;
                              sub_C8D5F0((__int64)&v121, v64, v121.m128i_u32[2] + 1LL, 0x10u, v37, v65);
                              v66 = (__m128i *)v121.m128i_i64[0];
                              v72 = v121.m128i_u32[2];
                              v73 = v100;
                              v64 = v102;
                            }
                            v74 = &v66[v72];
                            v74->m128i_i64[0] = v71;
                            v74->m128i_i64[1] = v73;
                            v68 = ++v121.m128i_i32[2];
                            *(_DWORD *)(v71 + 72) = v75;
                            v66 = (__m128i *)v121.m128i_i64[0];
                          }
                        }
                        while ( v68 );
                        v33 = v70;
                        v43 = v69;
                        v42 = v104;
                        v44 = v108;
                        v48 = v106;
                        *(_DWORD *)(v104 + 116) = 0;
                        *(_BYTE *)(v104 + 112) = 1;
                        if ( v66 != v64 )
                          _libc_free((unsigned __int64)v66);
                      }
LABEL_71:
                      if ( *(_DWORD *)(v48 + 72) >= *(_DWORD *)(v44 + 72)
                        && *(_DWORD *)(v48 + 76) <= *(_DWORD *)(v44 + 76) )
                      {
                        goto LABEL_66;
                      }
                    }
                    else
                    {
                      do
                      {
                        v50 = v48;
                        v48 = *(_QWORD *)(v48 + 8);
                      }
                      while ( v48 && *(_DWORD *)(v44 + 16) <= *(_DWORD *)(v48 + 16) );
                      if ( v44 == v50 )
                      {
LABEL_66:
                        if ( ++v33 != v111 )
                          continue;
                        a2 = v42;
LABEL_68:
                        v113 -= 2;
LABEL_19:
                        v6 = v113;
                        if ( v113 == v112 )
                        {
                          v99 = 0;
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
          sub_37F26B0(v43, v33);
          goto LABEL_66;
        }
        sub_37F2320(a1, v35);
        v87 = *(_DWORD *)(a1 + 24);
        if ( v87 )
        {
          v88 = v87 - 1;
          v37 = *(_QWORD *)(a1 + 8);
          v89 = 0;
          v90 = v88 & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
          v80 = *(_DWORD *)(a1 + 16) + 1;
          v91 = 1;
          v39 = v37 + 56LL * v90;
          v92 = *(_QWORD *)v39;
          if ( v103 != *(_QWORD *)v39 )
          {
            while ( v92 != -4096 )
            {
              if ( v92 == -8192 && !v89 )
                v89 = v39;
              v94 = v91 + 1;
              v95 = v88 & (v90 + v91);
              v90 = v95;
              v39 = v37 + 56 * v95;
              v92 = *(_QWORD *)v39;
              if ( v103 == *(_QWORD *)v39 )
                goto LABEL_117;
              v91 = v94;
            }
            if ( v89 )
              v39 = v89;
          }
          goto LABEL_117;
        }
LABEL_159:
        ++*(_DWORD *)(a1 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    sub_37F2320(a1, 2 * v35);
    v81 = *(_DWORD *)(a1 + 24);
    if ( v81 )
    {
      v82 = v81 - 1;
      v83 = *(_QWORD *)(a1 + 8);
      v84 = v82 & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
      v80 = *(_DWORD *)(a1 + 16) + 1;
      v39 = v83 + 56LL * v84;
      v37 = *(_QWORD *)v39;
      if ( v103 != *(_QWORD *)v39 )
      {
        v85 = 1;
        v86 = 0;
        while ( v37 != -4096 )
        {
          if ( !v86 && v37 == -8192 )
            v86 = v39;
          v96 = v85 + 1;
          v97 = v82 & (v84 + v85);
          v84 = v97;
          v39 = v83 + 56 * v97;
          v37 = *(_QWORD *)v39;
          if ( v103 == *(_QWORD *)v39 )
            goto LABEL_117;
          v85 = v96;
        }
        if ( v86 )
          v39 = v86;
      }
      goto LABEL_117;
    }
    goto LABEL_159;
  }
LABEL_21:
  if ( !v119 )
    _libc_free((unsigned __int64)v116);
  if ( v112 )
    j_j___libc_free_0((unsigned __int64)v112);
  return v99;
}
