// Function: sub_21064C0
// Address: 0x21064c0
//
_QWORD *__fastcall sub_21064C0(_QWORD *a1, __int64 a2, unsigned __int64 *a3)
{
  unsigned __int64 v3; // rcx
  __m128i *v4; // rax
  unsigned __int64 v5; // rsi
  _QWORD *v6; // r14
  _QWORD *v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int64 *v11; // rax
  char v12; // dl
  __int64 **v13; // r13
  __int64 *v14; // r14
  unsigned __int64 *v15; // rax
  __int64 v16; // rbx
  unsigned __int64 *v17; // r15
  __m128i *v18; // rsi
  _QWORD *v19; // r12
  unsigned __int64 *v21; // rdi
  unsigned __int64 *v22; // rcx
  unsigned __int64 *v23; // r13
  unsigned __int64 *v24; // r15
  __int64 v25; // rax
  unsigned __int64 v26; // rcx
  __int64 v27; // rdi
  unsigned int v28; // esi
  __int64 *v29; // rdx
  __int64 v30; // r11
  _QWORD *v31; // r12
  unsigned __int64 v32; // rdx
  _QWORD *v33; // rax
  _BOOL4 v34; // r10d
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // r12
  _QWORD *v38; // r14
  _QWORD *v39; // r13
  _QWORD *v40; // rax
  __int64 v41; // rcx
  __int64 v42; // rdx
  __int64 v43; // rbx
  __int64 v44; // r15
  __int64 v45; // rax
  __int64 v46; // rcx
  int v47; // r9d
  __int64 v48; // rdi
  unsigned int v49; // esi
  __int64 *v50; // rdx
  __int64 v51; // r10
  __int64 v52; // rax
  unsigned int v53; // edx
  __int64 v54; // rdx
  int v55; // edx
  _QWORD *v56; // rdx
  unsigned __int64 v57; // rcx
  unsigned __int64 v58; // rsi
  _QWORD *v59; // rax
  _BOOL4 v60; // r9d
  __int64 v61; // rax
  __int64 v62; // rax
  unsigned __int64 v63; // rdx
  int v64; // edx
  int v65; // r8d
  unsigned __int64 v66; // rdx
  int v67; // r8d
  __m128i *v68; // rdi
  int v69; // r12d
  unsigned __int32 v70; // edx
  __int64 v71; // r15
  __int64 v72; // rbx
  __int64 v73; // rdx
  __int64 v74; // r14
  __m128i *v75; // rdx
  int v76; // r13d
  __m128i *v77; // r14
  __int64 *v78; // rcx
  int v79; // r8d
  __int64 v80; // [rsp+8h] [rbp-408h]
  __int64 v82; // [rsp+30h] [rbp-3E0h]
  __int64 v83; // [rsp+30h] [rbp-3E0h]
  _QWORD *v84; // [rsp+40h] [rbp-3D0h]
  _BOOL4 v85; // [rsp+40h] [rbp-3D0h]
  _QWORD *v86; // [rsp+40h] [rbp-3D0h]
  _QWORD *v88; // [rsp+50h] [rbp-3C0h]
  char v89; // [rsp+58h] [rbp-3B8h]
  unsigned __int64 v90; // [rsp+58h] [rbp-3B8h]
  unsigned __int64 v91; // [rsp+58h] [rbp-3B8h]
  _BOOL4 v92; // [rsp+58h] [rbp-3B8h]
  unsigned __int64 v93; // [rsp+58h] [rbp-3B8h]
  __int64 v94; // [rsp+58h] [rbp-3B8h]
  __int64 v95; // [rsp+60h] [rbp-3B0h]
  _QWORD *v96; // [rsp+60h] [rbp-3B0h]
  _QWORD *v97; // [rsp+60h] [rbp-3B0h]
  __int64 v98; // [rsp+60h] [rbp-3B0h]
  _QWORD *v99; // [rsp+60h] [rbp-3B0h]
  __int64 **v100; // [rsp+68h] [rbp-3A8h]
  _QWORD *v101; // [rsp+68h] [rbp-3A8h]
  _QWORD *v102; // [rsp+68h] [rbp-3A8h]
  __int64 v103; // [rsp+70h] [rbp-3A0h] BYREF
  unsigned __int64 v104; // [rsp+78h] [rbp-398h] BYREF
  const __m128i *v105; // [rsp+80h] [rbp-390h] BYREF
  __m128i *v106; // [rsp+88h] [rbp-388h]
  const __m128i *v107; // [rsp+90h] [rbp-380h]
  __int64 v108; // [rsp+A0h] [rbp-370h] BYREF
  unsigned __int64 *v109; // [rsp+A8h] [rbp-368h]
  unsigned __int64 *v110; // [rsp+B0h] [rbp-360h]
  __int64 v111; // [rsp+B8h] [rbp-358h]
  int v112; // [rsp+C0h] [rbp-350h]
  _BYTE v113[264]; // [rsp+C8h] [rbp-348h] BYREF
  __m128i v114; // [rsp+1D0h] [rbp-240h] BYREF
  __m128i v115; // [rsp+1E0h] [rbp-230h] BYREF

  v115 = (__m128i)(unsigned __int64)a3;
  v3 = *a3;
  v109 = (unsigned __int64 *)v113;
  v110 = (unsigned __int64 *)v113;
  v105 = 0;
  v106 = 0;
  v107 = 0;
  v108 = 0;
  v111 = 32;
  v112 = 0;
  v114 = (__m128i)v3;
  sub_2106160(&v105, 0, &v114);
  v4 = v106;
  v88 = a1 + 1;
  while ( 1 )
  {
    v5 = v4[-2].m128i_u64[0];
    v6 = v88;
    v103 = v5;
    v104 = v4[-2].m128i_u64[1];
    v95 = v4[-1].m128i_i64[0];
    v82 = v4[-1].m128i_i64[1];
    v7 = (_QWORD *)a1[2];
    if ( !v7 )
      goto LABEL_9;
    do
    {
      while ( 1 )
      {
        v8 = v7[2];
        v9 = v7[3];
        if ( v7[4] >= v5 )
          break;
        v7 = (_QWORD *)v7[3];
        if ( !v9 )
          goto LABEL_7;
      }
      v6 = v7;
      v7 = (_QWORD *)v7[2];
    }
    while ( v8 );
LABEL_7:
    if ( v88 == v6 || v6[4] > v5 )
    {
LABEL_9:
      v114.m128i_i64[0] = (__int64)&v103;
      v10 = sub_21063F0(a1, v6, (unsigned __int64 **)&v114);
      v5 = v103;
      v6 = (_QWORD *)v10;
    }
    v11 = v109;
    if ( v110 == v109 )
    {
      v21 = &v109[HIDWORD(v111)];
      if ( v109 == v21 )
        goto LABEL_133;
      v22 = 0;
      do
      {
        if ( *v11 == v5 )
          goto LABEL_12;
        if ( *v11 == -2 )
          v22 = v11;
        ++v11;
      }
      while ( v21 != v11 );
      if ( !v22 )
      {
LABEL_133:
        if ( HIDWORD(v111) >= (unsigned int)v111 )
          goto LABEL_11;
        ++HIDWORD(v111);
        *v21 = v5;
        ++v108;
      }
      else
      {
        *v22 = v5;
        --v112;
        ++v108;
      }
LABEL_53:
      if ( *(_QWORD *)(v103 + 88) == *(_QWORD *)(v103 + 96) )
        goto LABEL_12;
      v101 = v6 + 6;
      v23 = *(unsigned __int64 **)(v103 + 96);
      v24 = *(unsigned __int64 **)(v103 + 88);
      while ( 1 )
      {
        v25 = *(unsigned int *)(a2 + 48);
        if ( !(_DWORD)v25 )
          goto LABEL_148;
        v26 = *v24;
        v27 = *(_QWORD *)(a2 + 32);
        v28 = (v25 - 1) & (((unsigned int)*v24 >> 9) ^ ((unsigned int)*v24 >> 4));
        v29 = (__int64 *)(v27 + 16LL * v28);
        v30 = *v29;
        if ( *v24 != *v29 )
        {
          v64 = 1;
          while ( v30 != -8 )
          {
            v65 = v64 + 1;
            v28 = (v25 - 1) & (v28 + v64);
            v29 = (__int64 *)(v27 + 16LL * v28);
            v30 = *v29;
            if ( v26 == *v29 )
              goto LABEL_57;
            v64 = v65;
          }
LABEL_148:
          BUG();
        }
LABEL_57:
        if ( v29 == (__int64 *)(v27 + 16 * v25) )
          goto LABEL_148;
        if ( *(_QWORD *)(v29[1] + 8) != v95 )
        {
          v31 = (_QWORD *)v6[7];
          if ( !v31 )
          {
            v31 = v6 + 6;
            if ( v101 == (_QWORD *)v6[8] )
            {
              v31 = v6 + 6;
              v34 = 1;
              goto LABEL_68;
            }
LABEL_73:
            v91 = *v24;
            v36 = sub_220EF80(v31);
            v26 = v91;
            if ( *(_QWORD *)(v36 + 32) >= v91 )
              goto LABEL_69;
            v34 = 1;
            if ( v101 == v31 )
              goto LABEL_68;
            goto LABEL_75;
          }
          while ( 1 )
          {
            v32 = v31[4];
            v33 = (_QWORD *)v31[3];
            if ( v26 < v32 )
              v33 = (_QWORD *)v31[2];
            if ( !v33 )
              break;
            v31 = v33;
          }
          if ( v26 < v32 )
          {
            if ( v31 != (_QWORD *)v6[8] )
              goto LABEL_73;
LABEL_67:
            v34 = 1;
            if ( v101 == v31 )
            {
LABEL_68:
              v85 = v34;
              v90 = v26;
              v35 = sub_22077B0(40);
              *(_QWORD *)(v35 + 32) = v90;
              sub_220F040(v85, v35, v31, v101);
              ++v6[10];
              goto LABEL_69;
            }
LABEL_75:
            v34 = v26 < v31[4];
            goto LABEL_68;
          }
          if ( v32 < v26 )
            goto LABEL_67;
        }
LABEL_69:
        if ( v23 == ++v24 )
          goto LABEL_12;
      }
    }
LABEL_11:
    sub_16CCBA0((__int64)&v108, v5);
    if ( v12 )
      goto LABEL_53;
LABEL_12:
    v89 = 0;
    v100 = *(__int64 ***)(v95 + 32);
    if ( v100 == *(__int64 ***)(v95 + 24) )
      goto LABEL_76;
    v84 = v6;
    v13 = *(__int64 ***)(v95 + 24);
    do
    {
      while ( 1 )
      {
        v14 = *v13;
        v15 = v109;
        v16 = **v13;
        if ( v110 == v109 )
        {
          v17 = &v109[HIDWORD(v111)];
          if ( v109 == v17 )
          {
            v63 = (unsigned __int64)v109;
          }
          else
          {
            do
            {
              if ( v16 == *v15 )
                break;
              ++v15;
            }
            while ( v17 != v15 );
            v63 = (unsigned __int64)&v109[HIDWORD(v111)];
          }
        }
        else
        {
          v17 = &v110[(unsigned int)v111];
          v15 = sub_16CC9F0((__int64)&v108, **v13);
          if ( v16 == *v15 )
          {
            v63 = (unsigned __int64)(v110 == v109 ? &v110[HIDWORD(v111)] : &v110[(unsigned int)v111]);
          }
          else
          {
            if ( v110 != v109 )
            {
              v15 = &v110[(unsigned int)v111];
              goto LABEL_19;
            }
            v15 = &v110[HIDWORD(v111)];
            v63 = (unsigned __int64)v15;
          }
        }
        while ( (unsigned __int64 *)v63 != v15 && *v15 >= 0xFFFFFFFFFFFFFFFELL )
          ++v15;
LABEL_19:
        if ( v15 == v17 )
          break;
        if ( v100 == ++v13 )
          goto LABEL_25;
      }
      v114.m128i_i64[0] = v16;
      v115.m128i_i64[0] = (__int64)v14;
      v18 = v106;
      v114.m128i_i64[1] = v103;
      v115.m128i_i64[1] = v95;
      if ( v106 == v107 )
      {
        sub_2106160(&v105, v106, &v114);
      }
      else
      {
        if ( v106 )
        {
          *v106 = _mm_load_si128(&v114);
          v18[1] = _mm_load_si128(&v115);
          v18 = v106;
        }
        v106 = v18 + 2;
      }
      v89 = 1;
      ++v13;
    }
    while ( v100 != v13 );
LABEL_25:
    v6 = v84;
    if ( v89 )
      goto LABEL_26;
LABEL_76:
    if ( !v104 )
      break;
    v37 = v6[8];
    v38 = v6 + 6;
    v39 = v88;
    v40 = (_QWORD *)a1[2];
    if ( !v40 )
      goto LABEL_84;
    do
    {
      while ( 1 )
      {
        v41 = v40[2];
        v42 = v40[3];
        if ( v40[4] >= v104 )
          break;
        v40 = (_QWORD *)v40[3];
        if ( !v42 )
          goto LABEL_82;
      }
      v39 = v40;
      v40 = (_QWORD *)v40[2];
    }
    while ( v41 );
LABEL_82:
    if ( v39 == v88 || v39[4] > v104 )
    {
LABEL_84:
      v114.m128i_i64[0] = (__int64)&v104;
      v39 = (_QWORD *)sub_21063F0(a1, v39, (unsigned __int64 **)&v114);
    }
    v102 = v39 + 6;
    if ( v38 != (_QWORD *)v37 )
    {
      v43 = v82;
      v44 = a2;
      do
      {
        v45 = *(unsigned int *)(v44 + 48);
        if ( !(_DWORD)v45 )
          goto LABEL_106;
        v46 = *(_QWORD *)(v37 + 32);
        v47 = v45 - 1;
        v48 = *(_QWORD *)(v44 + 32);
        v49 = (v45 - 1) & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
        v50 = (__int64 *)(v48 + 16LL * v49);
        v51 = *v50;
        if ( v46 != *v50 )
        {
          v55 = 1;
          while ( v51 != -8 )
          {
            v79 = v55 + 1;
            v49 = v47 & (v55 + v49);
            v50 = (__int64 *)(v48 + 16LL * v49);
            v51 = *v50;
            if ( v46 == *v50 )
              goto LABEL_89;
            v55 = v79;
          }
          goto LABEL_106;
        }
LABEL_89:
        if ( v50 == (__int64 *)(v48 + 16 * v45) )
          goto LABEL_106;
        v52 = v50[1];
        if ( !v43 || !v52 || v43 == v52 )
          goto LABEL_106;
        if ( v43 == *(_QWORD *)(v52 + 8) )
          goto LABEL_102;
        if ( v52 == *(_QWORD *)(v43 + 8) || *(_DWORD *)(v43 + 16) >= *(_DWORD *)(v52 + 16) )
          goto LABEL_106;
        if ( !*(_BYTE *)(v44 + 72) )
        {
          v53 = *(_DWORD *)(v44 + 76) + 1;
          *(_DWORD *)(v44 + 76) = v53;
          if ( v53 <= 0x20 )
          {
            do
            {
              v54 = v52;
              v52 = *(_QWORD *)(v52 + 8);
            }
            while ( v52 && *(_DWORD *)(v43 + 16) <= *(_DWORD *)(v52 + 16) );
            if ( v43 == v54 )
              goto LABEL_102;
            goto LABEL_106;
          }
          v66 = *(_QWORD *)(v44 + 56);
          v114.m128i_i32[3] = 32;
          v114.m128i_i64[0] = (__int64)&v115;
          if ( v66 )
          {
            v67 = 1;
            v115 = (__m128i)__PAIR128__(*(_QWORD *)(v66 + 24), v66);
            v68 = &v115;
            v114.m128i_i32[2] = 1;
            v94 = v37;
            v69 = 1;
            *(_DWORD *)(v66 + 48) = 0;
            v70 = 1;
            v83 = v44;
            v71 = v43;
            v99 = v38;
            v86 = v39;
            do
            {
              v76 = v69++;
              v77 = &v68[v70 - 1];
              v78 = (__int64 *)v77->m128i_i64[1];
              if ( v78 == *(__int64 **)(v77->m128i_i64[0] + 32) )
              {
                --v70;
                *(_DWORD *)(v77->m128i_i64[0] + 52) = v76;
                v114.m128i_i32[2] = v70;
              }
              else
              {
                v72 = *v78;
                v77->m128i_i64[1] = (__int64)(v78 + 1);
                v73 = v114.m128i_u32[2];
                v74 = *(_QWORD *)(v72 + 24);
                if ( v114.m128i_i32[2] >= (unsigned __int32)v114.m128i_i32[3] )
                {
                  v80 = v52;
                  sub_16CD150((__int64)&v114, &v115, 0, 16, v67, v47);
                  v68 = (__m128i *)v114.m128i_i64[0];
                  v73 = v114.m128i_u32[2];
                  v52 = v80;
                }
                v75 = &v68[v73];
                v75->m128i_i64[0] = v72;
                v75->m128i_i64[1] = v74;
                v70 = ++v114.m128i_i32[2];
                *(_DWORD *)(v72 + 48) = v76;
                v68 = (__m128i *)v114.m128i_i64[0];
              }
            }
            while ( v70 );
            v43 = v71;
            v44 = v83;
            v38 = v99;
            v37 = v94;
            v39 = v86;
            *(_DWORD *)(v83 + 76) = 0;
            *(_BYTE *)(v83 + 72) = 1;
            if ( v68 != &v115 )
            {
              v98 = v52;
              _libc_free((unsigned __int64)v68);
              v52 = v98;
            }
          }
        }
        if ( *(_DWORD *)(v52 + 48) >= *(_DWORD *)(v43 + 48) )
        {
          if ( *(_DWORD *)(v52 + 52) <= *(_DWORD *)(v43 + 52) )
            goto LABEL_102;
          v56 = (_QWORD *)v39[7];
          if ( !v56 )
          {
LABEL_125:
            v56 = v102;
            if ( (_QWORD *)v39[8] == v102 )
            {
              v56 = v102;
              v60 = 1;
            }
            else
            {
              v57 = *(_QWORD *)(v37 + 32);
LABEL_117:
              v93 = v57;
              v97 = v56;
              v62 = sub_220EF80(v56);
              v57 = v93;
              v56 = v97;
              if ( v93 <= *(_QWORD *)(v62 + 32) )
                goto LABEL_102;
              v60 = 1;
              if ( v97 != v102 )
LABEL_119:
                v60 = v57 < v56[4];
            }
LABEL_115:
            v92 = v60;
            v96 = v56;
            v61 = sub_22077B0(40);
            *(_QWORD *)(v61 + 32) = *(_QWORD *)(v37 + 32);
            sub_220F040(v92, v61, v96, v102);
            ++v39[10];
            goto LABEL_102;
          }
          goto LABEL_107;
        }
LABEL_106:
        v56 = (_QWORD *)v39[7];
        if ( !v56 )
          goto LABEL_125;
LABEL_107:
        v57 = *(_QWORD *)(v37 + 32);
        while ( 1 )
        {
          v58 = v56[4];
          v59 = (_QWORD *)v56[3];
          if ( v57 < v58 )
            v59 = (_QWORD *)v56[2];
          if ( !v59 )
            break;
          v56 = v59;
        }
        if ( v57 < v58 )
        {
          if ( v56 != (_QWORD *)v39[8] )
            goto LABEL_117;
LABEL_114:
          v60 = 1;
          if ( v56 != v102 )
            goto LABEL_119;
          goto LABEL_115;
        }
        if ( v57 > v58 )
          goto LABEL_114;
LABEL_102:
        v37 = sub_220EF30(v37);
      }
      while ( v38 != (_QWORD *)v37 );
    }
    v106 -= 2;
LABEL_26:
    v4 = v106;
    if ( v106 == v105 )
    {
      v19 = 0;
      goto LABEL_28;
    }
  }
  v19 = v6 + 5;
LABEL_28:
  if ( v110 != v109 )
    _libc_free((unsigned __int64)v110);
  if ( v105 )
    j_j___libc_free_0(v105, (char *)v107 - (char *)v105);
  return v19;
}
