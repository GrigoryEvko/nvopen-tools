// Function: sub_FA30D0
// Address: 0xfa30d0
//
__int64 __fastcall sub_FA30D0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  __int64 v6; // r12
  unsigned int v7; // eax
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 *v10; // rax
  __int64 v11; // r15
  __int64 v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 *v16; // rdi
  int v17; // esi
  unsigned int v18; // edx
  __int64 *v19; // r13
  __int64 v20; // r15
  unsigned __int64 v21; // rax
  unsigned int v22; // edi
  unsigned int v23; // edx
  __int64 v24; // rsi
  unsigned __int64 v25; // rax
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  int v28; // esi
  unsigned int v29; // edx
  __int64 v30; // rax
  int *v31; // rdx
  int v32; // eax
  __int64 v33; // rsi
  __int64 v34; // rax
  unsigned int v35; // ebx
  unsigned int v36; // r13d
  __int64 v37; // rax
  __int64 v38; // r14
  unsigned int v39; // ebx
  __int64 *v40; // rbx
  int v41; // r15d
  __int64 v42; // r13
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 *v45; // rbx
  __int64 *v46; // r13
  int v47; // esi
  __int64 *v48; // r8
  unsigned int v49; // ecx
  __int64 *v50; // rax
  __int64 v51; // r9
  __int64 v52; // rdx
  unsigned int v53; // esi
  __int64 v54; // rdx
  unsigned int v56; // esi
  unsigned int v57; // esi
  unsigned int v58; // edx
  unsigned int v59; // edx
  unsigned int v60; // edi
  __int64 v61; // rax
  __int64 v62; // r12
  unsigned __int64 v63; // rdx
  unsigned int v64; // edx
  unsigned int v65; // eax
  unsigned int v66; // edi
  _QWORD *v67; // rax
  __int64 v68; // rdx
  unsigned int v69; // eax
  unsigned int v70; // ecx
  unsigned int v71; // r8d
  __int64 v72; // rax
  int v73; // r10d
  unsigned int v74; // eax
  int v75; // r13d
  __int64 v76; // r10
  int v77; // r11d
  __int64 v78; // r10
  __int64 v79; // rdx
  __int64 v80; // rbx
  __int64 v81; // r12
  __int64 v82; // rax
  __int64 v83; // r14
  unsigned __int8 *v84; // rdx
  unsigned __int8 *i; // rdi
  __int64 v86; // rsi
  __int64 v87; // rdx
  __int64 v88; // r15
  __int64 v89; // r12
  __int64 v90; // r13
  __int64 v91; // rax
  __int64 v92; // r14
  unsigned int v93; // ecx
  __int64 v94; // rax
  void **v95; // r15
  __int64 v97; // [rsp+8h] [rbp-258h]
  unsigned int v99; // [rsp+2Ch] [rbp-234h]
  __int64 v100; // [rsp+30h] [rbp-230h]
  unsigned __int8 v101; // [rsp+38h] [rbp-228h]
  __int64 v102; // [rsp+40h] [rbp-220h]
  __int64 v103; // [rsp+40h] [rbp-220h]
  __int64 *v105; // [rsp+48h] [rbp-218h]
  int v106; // [rsp+48h] [rbp-218h]
  __int64 v107; // [rsp+58h] [rbp-208h] BYREF
  __m128i v108; // [rsp+60h] [rbp-200h] BYREF
  __int64 v109; // [rsp+70h] [rbp-1F0h] BYREF
  __int64 v110; // [rsp+78h] [rbp-1E8h]
  __int64 v111; // [rsp+80h] [rbp-1E0h]
  unsigned __int64 v112; // [rsp+90h] [rbp-1D0h] BYREF
  unsigned int v113; // [rsp+98h] [rbp-1C8h]
  __int64 v114; // [rsp+A0h] [rbp-1C0h] BYREF
  unsigned int v115; // [rsp+A8h] [rbp-1B8h]
  __int64 v116; // [rsp+B0h] [rbp-1B0h] BYREF
  char *v117; // [rsp+B8h] [rbp-1A8h]
  char v118; // [rsp+C8h] [rbp-198h] BYREF
  char v119; // [rsp+E8h] [rbp-178h]
  char v120; // [rsp+F0h] [rbp-170h]
  __int64 *v121; // [rsp+100h] [rbp-160h] BYREF
  __int64 v122; // [rsp+108h] [rbp-158h]
  _BYTE v123[64]; // [rsp+110h] [rbp-150h] BYREF
  __int64 *v124; // [rsp+150h] [rbp-110h] BYREF
  __int64 v125; // [rsp+158h] [rbp-108h]
  _BYTE v126[64]; // [rsp+160h] [rbp-100h] BYREF
  __int64 v127; // [rsp+1A0h] [rbp-C0h] BYREF
  __int64 v128; // [rsp+1A8h] [rbp-B8h]
  __int64 *v129; // [rsp+1B0h] [rbp-B0h] BYREF
  unsigned int v130; // [rsp+1B8h] [rbp-A8h]
  char v131; // [rsp+230h] [rbp-30h] BYREF

  v6 = a1;
  v97 = **(_QWORD **)(a1 - 8);
  sub_9AC3E0((__int64)&v112, v97, a4, 0, a3, a1, 0, 1);
  v7 = sub_9AF930(v97, a4, 0, a3, a1, 0);
  v127 = 0;
  v99 = v7;
  v121 = (__int64 *)v123;
  v122 = 0x800000000LL;
  v10 = (__int64 *)&v129;
  v128 = 1;
  do
  {
    *v10 = -4096;
    v10 += 2;
  }
  while ( v10 != (__int64 *)&v131 );
  v11 = *(_QWORD *)(a1 - 8);
  v124 = (__int64 *)v126;
  v125 = 0x800000000LL;
  v102 = ((*(_DWORD *)(a1 + 4) & 0x7FFFFFFu) >> 1) - 1;
  if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFFu) >> 1 != 1 )
  {
    v12 = 0;
    v13 = v11;
    while ( 1 )
    {
      v14 = 32;
      if ( (_DWORD)v12 != -2 )
        v14 = 32LL * (unsigned int)(2 * v12 + 3);
      v15 = *(_QWORD *)(v13 + v14);
      v109 = v15;
      if ( a2 )
        break;
LABEL_12:
      v20 = *(_QWORD *)(v13 + 32LL * (unsigned int)(2 * ++v12));
      if ( v113 <= 0x40 )
      {
        if ( (*(_QWORD *)(v20 + 24) & v112) != 0 )
          goto LABEL_22;
        if ( v115 > 0x40 )
        {
LABEL_15:
          if ( !(unsigned __int8)sub_C446F0(&v114, (__int64 *)(v20 + 24)) )
            goto LABEL_22;
          v21 = *(_QWORD *)(v20 + 24);
          goto LABEL_17;
        }
      }
      else
      {
        if ( (unsigned __int8)sub_C446A0((__int64 *)&v112, (__int64 *)(v20 + 24)) )
          goto LABEL_22;
        if ( v115 > 0x40 )
          goto LABEL_15;
      }
      v21 = *(_QWORD *)(v20 + 24);
      if ( (v114 & ~v21) != 0 )
        goto LABEL_22;
LABEL_17:
      v22 = *(_DWORD *)(v20 + 32);
      v23 = v22 + 1;
      v24 = 1LL << ((unsigned __int8)v22 - 1);
      if ( v22 > 0x40 )
      {
        if ( (*(_QWORD *)(v21 + 8LL * ((v22 - 1) >> 6)) & v24) != 0 )
          v58 = v23 - sub_C44500(v20 + 24);
        else
          v58 = v23 - sub_C444A0(v20 + 24);
        goto LABEL_87;
      }
      if ( (v24 & v21) == 0 )
      {
        v58 = 1;
        if ( v21 )
        {
          _BitScanReverse64(&v21, v21);
          v58 = 65 - (v21 ^ 0x3F);
        }
LABEL_87:
        if ( v99 >= v58 )
          goto LABEL_31;
LABEL_22:
        v26 = (unsigned int)v122;
        v27 = (unsigned int)v122 + 1LL;
        if ( v27 > HIDWORD(v122) )
        {
          sub_C8D5F0((__int64)&v121, v123, v27, 8u, v8, v9);
          v26 = (unsigned int)v122;
        }
        v121[v26] = v20;
        LODWORD(v122) = v122 + 1;
        if ( !a2 )
          goto LABEL_30;
        if ( (v128 & 1) != 0 )
        {
          v8 = (__int64)&v129;
          v28 = 7;
          goto LABEL_27;
        }
        v57 = v130;
        v8 = (__int64)v129;
        if ( !v130 )
        {
          v64 = v128;
          ++v127;
          v116 = 0;
          v65 = ((unsigned int)v128 >> 1) + 1;
          goto LABEL_98;
        }
        v28 = v130 - 1;
LABEL_27:
        v29 = v28 & (((unsigned int)v109 >> 9) ^ ((unsigned int)v109 >> 4));
        v30 = v8 + 16LL * v29;
        v9 = *(_QWORD *)v30;
        if ( *(_QWORD *)v30 == v109 )
        {
LABEL_28:
          v31 = (int *)(v30 + 8);
          v32 = *(_DWORD *)(v30 + 8) - 1;
          goto LABEL_29;
        }
        v75 = 1;
        v76 = 0;
        while ( v9 != -4096 )
        {
          if ( v9 == -8192 && !v76 )
            v76 = v30;
          v29 = v28 & (v75 + v29);
          v30 = v8 + 16LL * v29;
          v9 = *(_QWORD *)v30;
          if ( v109 == *(_QWORD *)v30 )
            goto LABEL_28;
          ++v75;
        }
        v64 = v128;
        v66 = 24;
        v57 = 8;
        if ( !v76 )
          v76 = v30;
        ++v127;
        v116 = v76;
        v65 = ((unsigned int)v128 >> 1) + 1;
        if ( (v128 & 1) == 0 )
        {
          v57 = v130;
LABEL_98:
          v66 = 3 * v57;
        }
        if ( 4 * v65 >= v66 )
        {
          v57 *= 2;
        }
        else if ( v57 - HIDWORD(v128) - v65 > v57 >> 3 )
        {
LABEL_101:
          v67 = (_QWORD *)v116;
          LODWORD(v128) = (2 * (v64 >> 1) + 2) | v64 & 1;
          if ( *(_QWORD *)v116 != -4096 )
            --HIDWORD(v128);
          v68 = v109;
          *(_DWORD *)(v116 + 8) = 0;
          *v67 = v68;
          v31 = (int *)(v67 + 1);
          v32 = -1;
LABEL_29:
          *v31 = v32;
LABEL_30:
          v13 = *(_QWORD *)(a1 - 8);
          goto LABEL_31;
        }
        sub_F5E3E0((__int64)&v127, v57);
        sub_F9BF10((__int64)&v127, &v109, &v116);
        v64 = v128;
        goto LABEL_101;
      }
      if ( !v22 )
      {
        v58 = 1;
        goto LABEL_87;
      }
      v25 = ~(v21 << (64 - (unsigned __int8)v22));
      if ( !v25 )
      {
        v58 = v22 - 63;
        goto LABEL_87;
      }
      _BitScanReverse64(&v25, v25);
      if ( v99 < v23 - ((unsigned int)v25 ^ 0x3F) )
        goto LABEL_22;
LABEL_31:
      if ( v12 == v102 )
      {
        v6 = a1;
        v11 = v13;
        goto LABEL_33;
      }
    }
    if ( (v128 & 1) != 0 )
    {
      v16 = (__int64 *)&v129;
      v17 = 7;
    }
    else
    {
      v56 = v130;
      v16 = v129;
      if ( !v130 )
      {
        v59 = v128;
        ++v127;
        v116 = 0;
        v60 = ((unsigned int)v128 >> 1) + 1;
        goto LABEL_90;
      }
      v17 = v130 - 1;
    }
    v18 = v17 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
    v19 = &v16[2 * v18];
    v8 = *v19;
    if ( v15 == *v19 )
    {
LABEL_11:
      ++*((_DWORD *)v19 + 2);
      v13 = *(_QWORD *)(a1 - 8);
      goto LABEL_12;
    }
    v73 = 1;
    v9 = 0;
    while ( v8 != -4096 )
    {
      if ( v8 == -8192 && !v9 )
        v9 = (__int64)v19;
      v18 = v17 & (v73 + v18);
      v19 = &v16[2 * v18];
      v8 = *v19;
      if ( v15 == *v19 )
        goto LABEL_11;
      ++v73;
    }
    v59 = v128;
    if ( !v9 )
      v9 = (__int64)v19;
    ++v127;
    v116 = v9;
    v60 = ((unsigned int)v128 >> 1) + 1;
    if ( (v128 & 1) != 0 )
    {
      v8 = 24;
      v56 = 8;
      if ( 4 * v60 >= 0x18 )
        goto LABEL_121;
      goto LABEL_91;
    }
    v56 = v130;
LABEL_90:
    v8 = 3 * v56;
    if ( (unsigned int)v8 <= 4 * v60 )
    {
LABEL_121:
      v56 *= 2;
      goto LABEL_122;
    }
LABEL_91:
    if ( v56 - HIDWORD(v128) - v60 > v56 >> 3 )
    {
LABEL_92:
      v19 = (__int64 *)v116;
      LODWORD(v128) = (2 * (v59 >> 1) + 2) | v59 & 1;
      if ( *(_QWORD *)v116 != -4096 )
        --HIDWORD(v128);
      *(_QWORD *)v116 = v15;
      *((_DWORD *)v19 + 2) = 0;
      v61 = (unsigned int)v125;
      v62 = v109;
      v63 = (unsigned int)v125 + 1LL;
      if ( v63 > HIDWORD(v125) )
      {
        sub_C8D5F0((__int64)&v124, v126, v63, 8u, v8, v9);
        v61 = (unsigned int)v125;
      }
      v124[v61] = v62;
      LODWORD(v125) = v125 + 1;
      goto LABEL_11;
    }
LABEL_122:
    sub_F5E3E0((__int64)&v127, v56);
    sub_F9BF10((__int64)&v127, &v109, &v116);
    v15 = v109;
    v59 = v128;
    goto LABEL_92;
  }
LABEL_33:
  v33 = 1;
  v34 = sub_AA5030(*(_QWORD *)(v11 + 32), 1);
  if ( !v34 )
    BUG();
  v35 = v113;
  v36 = *(unsigned __int8 *)(v34 - 24);
  v37 = v112;
  LODWORD(v110) = v113;
  if ( v113 <= 0x40 )
    goto LABEL_35;
  v33 = (__int64)&v112;
  sub_C43780((__int64)&v109, (const void **)&v112);
  if ( (unsigned int)v110 <= 0x40 )
  {
    v37 = v109;
LABEL_35:
    v38 = v114 | v37;
    goto LABEL_36;
  }
  v33 = (__int64)&v114;
  sub_C43BD0(&v109, &v114);
  v74 = v110;
  v38 = v109;
  LODWORD(v110) = 0;
  LODWORD(v117) = v74;
  v116 = v109;
  if ( v74 > 0x40 )
  {
    v39 = v35 - sub_C44630((__int64)&v116);
    if ( v38 )
    {
      j_j___libc_free_0_0(v38);
      if ( (unsigned int)v110 > 0x40 )
      {
        if ( v109 )
          j_j___libc_free_0_0(v109);
      }
    }
    goto LABEL_37;
  }
LABEL_36:
  v39 = v35 - sub_39FAC40(v38);
LABEL_37:
  if ( (_BYTE)v36 != 36 )
  {
    LOBYTE(v36) = v39 <= 0x3F && (_DWORD)v122 == 0;
    if ( (_BYTE)v36 )
    {
      v79 = 1LL << v39;
      v80 = ((*(_DWORD *)(v6 + 4) & 0x7FFFFFFu) >> 1) - 1;
      if ( v80 == v79 )
      {
        v33 = a2;
        sub_F9A0D0(v6, a2, 1);
        goto LABEL_64;
      }
      if ( v80 == v79 - 1 )
      {
        v83 = *(_QWORD *)(v97 + 8);
        v33 = *(_DWORD *)(v83 + 8) >> 8;
        if ( *(_DWORD *)(v83 + 8) <= 0x40FFu )
        {
          v84 = *(unsigned __int8 **)(a4 + 32);
          for ( i = &v84[*(_QWORD *)(a4 + 40)]; i != v84; ++v84 )
          {
            if ( (unsigned int)v33 <= *v84 )
            {
              v86 = 0;
              if ( (*(_DWORD *)(v6 + 4) & 0x7FFFFFFu) >> 1 != 1 )
              {
                v87 = *(_QWORD *)(v6 - 8);
                v101 = v36;
                v88 = 1;
                v100 = v6;
                v89 = 0;
                v103 = *(_QWORD *)(v97 + 8);
                v90 = v87;
                while ( 1 )
                {
                  v92 = *(_QWORD *)(v90 + 32LL * (unsigned int)(2 * v88));
                  if ( *(_DWORD *)(v92 + 32) <= 0x40u )
                  {
                    v91 = *(_QWORD *)(v92 + 24);
                  }
                  else
                  {
                    v106 = *(_DWORD *)(v92 + 32);
                    v93 = v106 - sub_C444A0(v92 + 24);
                    v91 = -1;
                    if ( v93 <= 0x40 )
                      v91 = **(_QWORD **)(v92 + 24);
                  }
                  v89 ^= v91;
                  if ( v80 == v88 )
                    break;
                  ++v88;
                }
                v86 = v89;
                v83 = v103;
                v36 = v101;
                v6 = v100;
              }
              v94 = sub_AD64C0(v83, v86, 0);
              v116 = v6;
              v95 = (void **)v94;
              v119 = 0;
              v120 = 0;
              sub_B540B0(&v116);
              v109 = sub_B543C0((__int64)&v116, 0);
              sub_B541D0((__int64)&v116, v95, *(_QWORD *)(*(_QWORD *)(v6 - 8) + 32LL), v109);
              sub_F9A0D0(v6, a2, 0);
              v33 = 0;
              v108.m128i_i64[0] = 0x100000000LL;
              sub_B543F0((__int64)&v116, 0, 0x100000000uLL);
              sub_F92F70((__int64)&v116, 0);
              goto LABEL_64;
            }
          }
        }
      }
      v36 = 0;
      goto LABEL_64;
    }
  }
  v36 = 0;
  if ( !(_DWORD)v122 )
    goto LABEL_64;
  v116 = v6;
  v119 = 0;
  v120 = 0;
  sub_B540B0(&v116);
  v40 = v121;
  v105 = &v121[(unsigned int)v122];
  if ( v105 != v121 )
  {
    do
    {
      v41 = -2;
      v42 = ((*(_DWORD *)(v6 + 4) & 0x7FFFFFFu) >> 1) - 1;
      sub_F90F90(v6, 0, v6, v42, *v40);
      v44 = 32;
      if ( v43 != v42 )
      {
        v41 = v43;
        if ( (unsigned int)v43 != 4294967294LL )
          v44 = 32LL * (unsigned int)(2 * v43 + 3);
      }
      ++v40;
      sub_AA5980(*(_QWORD *)(*(_QWORD *)(v6 - 8) + v44), *(_QWORD *)(v6 + 40), 0);
      v33 = v6;
      sub_B541A0((__int64)&v116, v6, v41);
    }
    while ( v105 != v40 );
  }
  if ( !a2 )
    goto LABEL_58;
  v45 = v124;
  v109 = 0;
  v110 = 0;
  v111 = 0;
  v46 = &v124[(unsigned int)v125];
  if ( v46 == v124 )
  {
    v54 = 0;
    v33 = 0;
    goto LABEL_56;
  }
  do
  {
    v52 = *v45;
    v107 = *v45;
    if ( (v128 & 1) != 0 )
    {
      v47 = 7;
      v48 = (__int64 *)&v129;
    }
    else
    {
      v53 = v130;
      v48 = v129;
      if ( !v130 )
      {
        v69 = v128;
        ++v127;
        v108.m128i_i64[0] = 0;
        v70 = ((unsigned int)v128 >> 1) + 1;
LABEL_108:
        v71 = 3 * v53;
        goto LABEL_109;
      }
      v47 = v130 - 1;
    }
    v49 = v47 & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
    v50 = &v48[2 * v49];
    v51 = *v50;
    if ( v52 == *v50 )
    {
LABEL_50:
      if ( *((_DWORD *)v50 + 2) )
        goto LABEL_51;
      goto LABEL_114;
    }
    v77 = 1;
    v78 = 0;
    while ( v51 != -4096 )
    {
      if ( v51 == -8192 && !v78 )
        v78 = (__int64)v50;
      v49 = v47 & (v77 + v49);
      v50 = &v48[2 * v49];
      v51 = *v50;
      if ( v52 == *v50 )
        goto LABEL_50;
      ++v77;
    }
    v71 = 24;
    v53 = 8;
    if ( !v78 )
      v78 = (__int64)v50;
    v69 = v128;
    ++v127;
    v108.m128i_i64[0] = v78;
    v70 = ((unsigned int)v128 >> 1) + 1;
    if ( (v128 & 1) == 0 )
    {
      v53 = v130;
      goto LABEL_108;
    }
LABEL_109:
    if ( v71 <= 4 * v70 )
    {
      sub_F5E3E0((__int64)&v127, 2 * v53);
LABEL_148:
      sub_F9BF10((__int64)&v127, &v107, &v108);
      v52 = v107;
      v69 = v128;
      goto LABEL_111;
    }
    if ( v53 - HIDWORD(v128) - v70 <= v53 >> 3 )
    {
      sub_F5E3E0((__int64)&v127, v53);
      goto LABEL_148;
    }
LABEL_111:
    LODWORD(v128) = (2 * (v69 >> 1) + 2) | v69 & 1;
    v72 = v108.m128i_i64[0];
    if ( *(_QWORD *)v108.m128i_i64[0] != -4096 )
      --HIDWORD(v128);
    *(_QWORD *)v108.m128i_i64[0] = v52;
    *(_DWORD *)(v72 + 8) = 0;
LABEL_114:
    v108.m128i_i64[0] = *(_QWORD *)(v6 + 40);
    v108.m128i_i64[1] = v107 | 4;
    sub_F9E360((__int64)&v109, &v108);
LABEL_51:
    ++v45;
  }
  while ( v46 != v45 );
  v33 = v109;
  v54 = (v110 - v109) >> 4;
LABEL_56:
  sub_FFB3D0(a2, v33, v54);
  if ( v109 )
  {
    v33 = v111 - v109;
    j_j___libc_free_0(v109, v111 - v109);
  }
LABEL_58:
  if ( v120 )
  {
    v81 = v116;
    v82 = sub_B53F50((__int64)&v116);
    v33 = 2;
    sub_B99FD0(v81, 2u, v82);
  }
  if ( v119 && v117 != &v118 )
    _libc_free(v117, v33);
  v36 = 1;
LABEL_64:
  if ( v124 != (__int64 *)v126 )
    _libc_free(v124, v33);
  if ( (v128 & 1) == 0 )
  {
    v33 = 16LL * v130;
    sub_C7D6A0((__int64)v129, v33, 8);
  }
  if ( v121 != (__int64 *)v123 )
    _libc_free(v121, v33);
  if ( v115 > 0x40 && v114 )
    j_j___libc_free_0_0(v114);
  if ( v113 > 0x40 && v112 )
    j_j___libc_free_0_0(v112);
  return v36;
}
