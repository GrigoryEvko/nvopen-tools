// Function: sub_13BFEC0
// Address: 0x13bfec0
//
_QWORD *__fastcall sub_13BFEC0(_QWORD *a1, __int64 a2, unsigned __int64 *a3)
{
  unsigned __int64 v3; // rcx
  __m128i *v4; // rax
  unsigned __int64 v5; // rsi
  _QWORD *v6; // rax
  _QWORD *v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int64 *v11; // rax
  char v12; // dl
  __int64 **v13; // r15
  unsigned __int64 *v14; // r12
  unsigned __int64 *v15; // rax
  __int64 *v16; // r14
  __int64 v17; // rbx
  unsigned __int64 *v18; // rdx
  __m128i *v19; // rsi
  _QWORD *v20; // r12
  unsigned __int64 *v22; // rdi
  unsigned __int64 *v23; // rcx
  __int64 v24; // rdi
  int v25; // r13d
  __int64 v26; // rbx
  unsigned int v27; // r15d
  unsigned __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rdi
  unsigned int v31; // esi
  __int64 *v32; // rdx
  __int64 v33; // r10
  _QWORD *v34; // rdx
  unsigned __int64 v35; // rsi
  _QWORD *v36; // rax
  _BOOL4 v37; // r9d
  __int64 v38; // rax
  _QWORD *v39; // r14
  __int64 v40; // r13
  _QWORD *v41; // r12
  _QWORD *v42; // rax
  __int64 v43; // rcx
  __int64 v44; // rdx
  __int64 v45; // rbx
  __int64 v46; // r15
  __int64 v47; // rax
  __int64 v48; // rsi
  __int64 v49; // rdi
  unsigned int v50; // ecx
  __int64 *v51; // rdx
  __int64 v52; // r10
  __int64 v53; // rax
  _QWORD *v54; // rdx
  unsigned __int64 v55; // rsi
  unsigned __int64 v56; // rcx
  _QWORD *v57; // rax
  _BOOL4 v58; // r9d
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rax
  int v62; // edx
  int v63; // r8d
  unsigned int v64; // edx
  __int64 v65; // rdx
  int v66; // edx
  int v67; // r8d
  unsigned __int64 v68; // rdx
  __m128i *v69; // rdi
  int v70; // r12d
  unsigned __int32 v71; // edx
  __int64 v72; // r15
  __int64 v73; // rbx
  __int64 v74; // rdx
  __int64 v75; // r14
  __m128i *v76; // rdx
  int v77; // r13d
  __m128i *v78; // r14
  __int64 *v79; // rcx
  _BOOL4 v81; // [rsp+20h] [rbp-3F0h]
  __int64 v82; // [rsp+20h] [rbp-3F0h]
  _QWORD *v83; // [rsp+28h] [rbp-3E8h]
  unsigned __int64 v84; // [rsp+28h] [rbp-3E8h]
  __int64 v85; // [rsp+30h] [rbp-3E0h]
  __int64 v86; // [rsp+30h] [rbp-3E0h]
  _QWORD *v88; // [rsp+48h] [rbp-3C8h]
  _QWORD *v89; // [rsp+50h] [rbp-3C0h]
  _QWORD *v90; // [rsp+50h] [rbp-3C0h]
  char v91; // [rsp+58h] [rbp-3B8h]
  unsigned __int64 v92; // [rsp+58h] [rbp-3B8h]
  _QWORD *v93; // [rsp+58h] [rbp-3B8h]
  _QWORD *v94; // [rsp+58h] [rbp-3B8h]
  __int64 v95; // [rsp+58h] [rbp-3B8h]
  __int64 v96; // [rsp+60h] [rbp-3B0h]
  _BOOL4 v97; // [rsp+60h] [rbp-3B0h]
  _QWORD *v98; // [rsp+60h] [rbp-3B0h]
  _QWORD *v99; // [rsp+60h] [rbp-3B0h]
  __int64 v100; // [rsp+60h] [rbp-3B0h]
  __int64 **v101; // [rsp+68h] [rbp-3A8h]
  _QWORD *v102; // [rsp+68h] [rbp-3A8h]
  _QWORD *v103; // [rsp+68h] [rbp-3A8h]
  __int64 v104; // [rsp+70h] [rbp-3A0h] BYREF
  unsigned __int64 v105; // [rsp+78h] [rbp-398h] BYREF
  const __m128i *v106; // [rsp+80h] [rbp-390h] BYREF
  __m128i *v107; // [rsp+88h] [rbp-388h]
  const __m128i *v108; // [rsp+90h] [rbp-380h]
  __int64 v109; // [rsp+A0h] [rbp-370h] BYREF
  unsigned __int64 *v110; // [rsp+A8h] [rbp-368h]
  unsigned __int64 *v111; // [rsp+B0h] [rbp-360h]
  __int64 v112; // [rsp+B8h] [rbp-358h]
  int v113; // [rsp+C0h] [rbp-350h]
  _BYTE v114[264]; // [rsp+C8h] [rbp-348h] BYREF
  __m128i v115; // [rsp+1D0h] [rbp-240h] BYREF
  __m128i v116; // [rsp+1E0h] [rbp-230h] BYREF

  v116 = (__m128i)(unsigned __int64)a3;
  v3 = *a3;
  v110 = (unsigned __int64 *)v114;
  v111 = (unsigned __int64 *)v114;
  v115 = (__m128i)v3;
  v106 = 0;
  v107 = 0;
  v108 = 0;
  v109 = 0;
  v112 = 32;
  v113 = 0;
  sub_13BFB60(&v106, 0, &v115);
  v4 = v107;
  v88 = a1 + 1;
  while ( 1 )
  {
    v5 = v4[-2].m128i_u64[0];
    v104 = v5;
    v105 = v4[-2].m128i_u64[1];
    v96 = v4[-1].m128i_i64[0];
    v85 = v4[-1].m128i_i64[1];
    v6 = (_QWORD *)a1[2];
    if ( !v6 )
    {
      v89 = v88;
LABEL_9:
      v115.m128i_i64[0] = (__int64)&v104;
      v10 = sub_13BFDF0(a1, v89, (unsigned __int64 **)&v115);
      v5 = v104;
      v89 = (_QWORD *)v10;
      goto LABEL_10;
    }
    v7 = v88;
    do
    {
      while ( 1 )
      {
        v8 = v6[2];
        v9 = v6[3];
        if ( v6[4] >= v5 )
          break;
        v6 = (_QWORD *)v6[3];
        if ( !v9 )
          goto LABEL_7;
      }
      v7 = v6;
      v6 = (_QWORD *)v6[2];
    }
    while ( v8 );
LABEL_7:
    v89 = v7;
    if ( v88 == v7 || v7[4] > v5 )
      goto LABEL_9;
LABEL_10:
    v11 = v110;
    if ( v111 == v110 )
    {
      v22 = &v110[HIDWORD(v112)];
      if ( v110 != v22 )
      {
        v23 = 0;
        do
        {
          if ( v5 == *v11 )
            goto LABEL_12;
          if ( *v11 == -2 )
            v23 = v11;
          ++v11;
        }
        while ( v22 != v11 );
        if ( v23 )
        {
          *v23 = v5;
          --v113;
          ++v109;
          goto LABEL_54;
        }
      }
      if ( HIDWORD(v112) < (unsigned int)v112 )
      {
        ++HIDWORD(v112);
        *v22 = v5;
        ++v109;
        goto LABEL_54;
      }
    }
    sub_16CCBA0(&v109, v5);
    if ( !v12 )
      goto LABEL_12;
LABEL_54:
    v24 = sub_157EBA0(v104);
    if ( v24 )
    {
      v25 = sub_15F4D60(v24);
      v26 = sub_157EBA0(v104);
      if ( v25 )
      {
        v27 = 0;
        v102 = v89 + 6;
        do
        {
          v28 = sub_15F4DF0(v26, v27);
          v29 = *(unsigned int *)(a2 + 48);
          if ( !(_DWORD)v29 )
            goto LABEL_150;
          v30 = *(_QWORD *)(a2 + 32);
          v31 = (v29 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
          v32 = (__int64 *)(v30 + 16LL * v31);
          v33 = *v32;
          if ( v28 != *v32 )
          {
            v66 = 1;
            while ( v33 != -8 )
            {
              v67 = v66 + 1;
              v31 = (v29 - 1) & (v66 + v31);
              v32 = (__int64 *)(v30 + 16LL * v31);
              v33 = *v32;
              if ( v28 == *v32 )
                goto LABEL_59;
              v66 = v67;
            }
LABEL_150:
            BUG();
          }
LABEL_59:
          if ( v32 == (__int64 *)(v30 + 16 * v29) )
            goto LABEL_150;
          if ( v96 != *(_QWORD *)(v32[1] + 8) )
          {
            v34 = (_QWORD *)v89[7];
            if ( !v34 )
            {
              v34 = v89 + 6;
              if ( (_QWORD *)v89[8] == v102 )
              {
                v34 = v89 + 6;
                v37 = 1;
                goto LABEL_70;
              }
LABEL_109:
              v84 = v28;
              v94 = v34;
              v60 = sub_220EF80(v34);
              v28 = v84;
              v34 = v94;
              if ( *(_QWORD *)(v60 + 32) >= v84 )
                goto LABEL_71;
              v37 = 1;
              if ( v94 == v102 )
                goto LABEL_70;
              goto LABEL_111;
            }
            while ( 1 )
            {
              v35 = v34[4];
              v36 = (_QWORD *)v34[3];
              if ( v28 < v35 )
                v36 = (_QWORD *)v34[2];
              if ( !v36 )
                break;
              v34 = v36;
            }
            if ( v28 < v35 )
            {
              if ( (_QWORD *)v89[8] != v34 )
                goto LABEL_109;
LABEL_69:
              v37 = 1;
              if ( v34 == v102 )
              {
LABEL_70:
                v81 = v37;
                v83 = v34;
                v92 = v28;
                v38 = sub_22077B0(40);
                *(_QWORD *)(v38 + 32) = v92;
                sub_220F040(v81, v38, v83, v102);
                ++v89[10];
                goto LABEL_71;
              }
LABEL_111:
              v37 = v28 < v34[4];
              goto LABEL_70;
            }
            if ( v35 < v28 )
              goto LABEL_69;
          }
LABEL_71:
          ++v27;
        }
        while ( v27 != v25 );
      }
    }
LABEL_12:
    v91 = 0;
    v13 = *(__int64 ***)(v96 + 24);
    v101 = *(__int64 ***)(v96 + 32);
    if ( v101 == v13 )
      goto LABEL_73;
    do
    {
      while ( 1 )
      {
        v16 = *v13;
        v15 = v110;
        v17 = **v13;
        if ( v111 != v110 )
          break;
        v18 = &v110[HIDWORD(v112)];
        if ( v110 == v18 )
        {
          v14 = v110;
        }
        else
        {
          do
          {
            if ( v17 == *v15 )
              break;
            ++v15;
          }
          while ( v18 != v15 );
          v14 = &v110[HIDWORD(v112)];
        }
LABEL_27:
        while ( v18 != v15 )
        {
          if ( *v15 < 0xFFFFFFFFFFFFFFFELL )
            goto LABEL_17;
          ++v15;
        }
        if ( v14 == v15 )
          goto LABEL_29;
LABEL_18:
        if ( v101 == ++v13 )
          goto LABEL_34;
      }
      v14 = &v111[(unsigned int)v112];
      v15 = (unsigned __int64 *)sub_16CC9F0(&v109, **v13);
      if ( v17 == *v15 )
      {
        if ( v111 == v110 )
          v18 = &v111[HIDWORD(v112)];
        else
          v18 = &v111[(unsigned int)v112];
        goto LABEL_27;
      }
      if ( v111 == v110 )
      {
        v18 = &v111[HIDWORD(v112)];
        v15 = v18;
        goto LABEL_27;
      }
      v15 = &v111[(unsigned int)v112];
LABEL_17:
      if ( v14 != v15 )
        goto LABEL_18;
LABEL_29:
      v115.m128i_i64[0] = v17;
      v116.m128i_i64[0] = (__int64)v16;
      v19 = v107;
      v115.m128i_i64[1] = v104;
      v116.m128i_i64[1] = v96;
      if ( v107 == v108 )
      {
        sub_13BFB60(&v106, v107, &v115);
      }
      else
      {
        if ( v107 )
        {
          *v107 = _mm_load_si128(&v115);
          v19[1] = _mm_load_si128(&v116);
          v19 = v107;
        }
        v107 = v19 + 2;
      }
      v91 = 1;
      ++v13;
    }
    while ( v101 != v13 );
LABEL_34:
    if ( v91 )
      goto LABEL_35;
LABEL_73:
    if ( !v105 )
      break;
    v39 = v88;
    v40 = v89[8];
    v41 = v89 + 6;
    v42 = (_QWORD *)a1[2];
    if ( !v42 )
      goto LABEL_81;
    do
    {
      while ( 1 )
      {
        v43 = v42[2];
        v44 = v42[3];
        if ( v42[4] >= v105 )
          break;
        v42 = (_QWORD *)v42[3];
        if ( !v44 )
          goto LABEL_79;
      }
      v39 = v42;
      v42 = (_QWORD *)v42[2];
    }
    while ( v43 );
LABEL_79:
    if ( v39 == v88 || v39[4] > v105 )
    {
LABEL_81:
      v115.m128i_i64[0] = (__int64)&v105;
      v39 = (_QWORD *)sub_13BFDF0(a1, v39, (unsigned __int64 **)&v115);
    }
    v103 = v39 + 6;
    if ( v41 != (_QWORD *)v40 )
    {
      v45 = v85;
      v46 = a2;
      do
      {
        v47 = *(unsigned int *)(v46 + 48);
        if ( (_DWORD)v47 )
        {
          v48 = *(_QWORD *)(v40 + 32);
          v49 = *(_QWORD *)(v46 + 32);
          v50 = (v47 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
          v51 = (__int64 *)(v49 + 16LL * v50);
          v52 = *v51;
          if ( v48 == *v51 )
          {
LABEL_86:
            if ( v51 != (__int64 *)(v49 + 16 * v47) )
            {
              v53 = v51[1];
              if ( v45 )
              {
                if ( v53 && v45 != v53 )
                {
                  if ( v45 == *(_QWORD *)(v53 + 8) )
                    goto LABEL_106;
                  if ( v53 != *(_QWORD *)(v45 + 8) && *(_DWORD *)(v45 + 16) < *(_DWORD *)(v53 + 16) )
                  {
                    if ( *(_BYTE *)(v46 + 72) )
                    {
                      if ( *(_DWORD *)(v53 + 48) >= *(_DWORD *)(v45 + 48) )
                        goto LABEL_95;
                    }
                    else
                    {
                      v64 = *(_DWORD *)(v46 + 76) + 1;
                      *(_DWORD *)(v46 + 76) = v64;
                      if ( v64 > 0x20 )
                      {
                        v68 = *(_QWORD *)(v46 + 56);
                        v115.m128i_i32[3] = 32;
                        v115.m128i_i64[0] = (__int64)&v116;
                        if ( v68 )
                        {
                          v116 = (__m128i)__PAIR128__(*(_QWORD *)(v68 + 24), v68);
                          v69 = &v116;
                          v115.m128i_i32[2] = 1;
                          v99 = v41;
                          v70 = 1;
                          *(_DWORD *)(v68 + 48) = 0;
                          v71 = 1;
                          v86 = v46;
                          v72 = v45;
                          v95 = v40;
                          v90 = v39;
                          do
                          {
                            while ( 1 )
                            {
                              v77 = v70++;
                              v78 = &v69[v71 - 1];
                              v79 = (__int64 *)v78->m128i_i64[1];
                              if ( v79 != *(__int64 **)(v78->m128i_i64[0] + 32) )
                                break;
                              --v71;
                              *(_DWORD *)(v78->m128i_i64[0] + 52) = v77;
                              v115.m128i_i32[2] = v71;
                              if ( !v71 )
                                goto LABEL_144;
                            }
                            v73 = *v79;
                            v78->m128i_i64[1] = (__int64)(v79 + 1);
                            v74 = v115.m128i_u32[2];
                            v75 = *(_QWORD *)(v73 + 24);
                            if ( v115.m128i_i32[2] >= (unsigned __int32)v115.m128i_i32[3] )
                            {
                              v82 = v53;
                              sub_16CD150(&v115, &v116, 0, 16);
                              v69 = (__m128i *)v115.m128i_i64[0];
                              v74 = v115.m128i_u32[2];
                              v53 = v82;
                            }
                            v76 = &v69[v74];
                            v76->m128i_i64[0] = v73;
                            v76->m128i_i64[1] = v75;
                            v71 = ++v115.m128i_i32[2];
                            *(_DWORD *)(v73 + 48) = v77;
                            v69 = (__m128i *)v115.m128i_i64[0];
                          }
                          while ( v71 );
LABEL_144:
                          v45 = v72;
                          v46 = v86;
                          v41 = v99;
                          v40 = v95;
                          v39 = v90;
                          *(_DWORD *)(v86 + 76) = 0;
                          *(_BYTE *)(v86 + 72) = 1;
                          if ( v69 != &v116 )
                          {
                            v100 = v53;
                            _libc_free((unsigned __int64)v69);
                            v53 = v100;
                          }
                        }
                        if ( *(_DWORD *)(v53 + 48) >= *(_DWORD *)(v45 + 48) )
                        {
LABEL_95:
                          if ( *(_DWORD *)(v53 + 52) <= *(_DWORD *)(v45 + 52) )
                            goto LABEL_106;
                        }
                      }
                      else
                      {
                        do
                        {
                          v65 = v53;
                          v53 = *(_QWORD *)(v53 + 8);
                        }
                        while ( v53 && *(_DWORD *)(v45 + 16) <= *(_DWORD *)(v53 + 16) );
                        if ( v45 == v65 )
                          goto LABEL_106;
                      }
                    }
                  }
                }
              }
            }
          }
          else
          {
            v62 = 1;
            while ( v52 != -8 )
            {
              v63 = v62 + 1;
              v50 = (v47 - 1) & (v62 + v50);
              v51 = (__int64 *)(v49 + 16LL * v50);
              v52 = *v51;
              if ( v48 == *v51 )
                goto LABEL_86;
              v62 = v63;
            }
          }
        }
        v54 = (_QWORD *)v39[7];
        if ( !v54 )
        {
          v54 = v103;
          if ( (_QWORD *)v39[8] == v103 )
          {
            v54 = v103;
            v58 = 1;
            goto LABEL_105;
          }
LABEL_113:
          v98 = v54;
          v61 = sub_220EF80(v54);
          v54 = v98;
          if ( *(_QWORD *)(v61 + 32) >= *(_QWORD *)(v40 + 32) )
            goto LABEL_106;
          v58 = 1;
          if ( v103 == v98 )
            goto LABEL_105;
          goto LABEL_115;
        }
        v55 = *(_QWORD *)(v40 + 32);
        while ( 1 )
        {
          v56 = v54[4];
          v57 = (_QWORD *)v54[3];
          if ( v55 < v56 )
            v57 = (_QWORD *)v54[2];
          if ( !v57 )
            break;
          v54 = v57;
        }
        if ( v55 < v56 )
        {
          if ( (_QWORD *)v39[8] != v54 )
            goto LABEL_113;
LABEL_104:
          v58 = 1;
          if ( v103 == v54 )
          {
LABEL_105:
            v93 = v54;
            v97 = v58;
            v59 = sub_22077B0(40);
            *(_QWORD *)(v59 + 32) = *(_QWORD *)(v40 + 32);
            sub_220F040(v97, v59, v93, v103);
            ++v39[10];
            goto LABEL_106;
          }
LABEL_115:
          v58 = *(_QWORD *)(v40 + 32) < v54[4];
          goto LABEL_105;
        }
        if ( v55 > v56 )
          goto LABEL_104;
LABEL_106:
        v40 = sub_220EF30(v40);
      }
      while ( v41 != (_QWORD *)v40 );
    }
    v107 -= 2;
LABEL_35:
    v4 = v107;
    if ( v107 == v106 )
    {
      v20 = 0;
      goto LABEL_37;
    }
  }
  v20 = v89 + 5;
LABEL_37:
  if ( v111 != v110 )
    _libc_free((unsigned __int64)v111);
  if ( v106 )
    j_j___libc_free_0(v106, (char *)v108 - (char *)v106);
  return v20;
}
