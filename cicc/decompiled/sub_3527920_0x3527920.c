// Function: sub_3527920
// Address: 0x3527920
//
__int64 __fastcall sub_3527920(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 *v3; // rdx
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rdx
  unsigned __int8 v8; // al
  __int64 v9; // rax
  __int64 v10; // rax
  _QWORD *v11; // r13
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // r9
  unsigned __int8 v15; // al
  __int64 v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // r12
  unsigned int v19; // r15d
  unsigned int v20; // r14d
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 v24; // r14
  __int64 v25; // rbx
  __int64 v26; // rbx
  __int64 v27; // rax
  unsigned __int8 v28; // dl
  __int64 v29; // rdi
  __int64 v30; // rdx
  __int64 v31; // rsi
  __int64 v32; // rdx
  __int64 v33; // rax
  unsigned int v34; // esi
  __int64 v35; // rcx
  char v37; // bl
  unsigned int v38; // esi
  __int64 v39; // rcx
  _QWORD *v41; // rdi
  __m128i *v42; // rax
  __m128i v43; // xmm0
  char *v44; // rsi
  __int64 v45; // rdi
  _DWORD *v46; // rdx
  _BYTE *v47; // rax
  __int64 v49; // rsi
  __int64 v50; // rsi
  _QWORD *v51; // rax
  __m128i *v52; // rdx
  __m128i si128; // xmm0
  const char *v54; // rax
  size_t v55; // rdx
  __int64 v56; // r8
  unsigned __int8 *v57; // rsi
  _BYTE *v58; // rdi
  unsigned __int64 v59; // rax
  void *v60; // rax
  int v61; // eax
  __int64 v62; // rax
  __int64 v63; // rax
  int v65; // r12d
  _QWORD *v66; // rax
  __m128i *v67; // rdx
  __int64 v68; // rdi
  __m128i v69; // xmm0
  unsigned int v70; // r12d
  __int64 v71; // rdi
  _BYTE *v72; // rax
  unsigned int v73; // r10d
  unsigned int v74; // esi
  int v75; // r12d
  unsigned __int64 v76; // r8
  __int64 v77; // rdx
  unsigned __int64 v78; // r8
  int v82; // r12d
  _QWORD *v83; // rax
  __m128i *v84; // rdx
  __int64 v85; // rdi
  __m128i v86; // xmm0
  unsigned int v87; // r12d
  __int64 v88; // rdi
  _BYTE *v89; // rax
  unsigned int v90; // r10d
  unsigned int v91; // esi
  int v92; // r12d
  unsigned __int64 v93; // r8
  __int64 v94; // rdx
  unsigned __int64 v95; // r8
  _QWORD *v98; // rdi
  __m128i *v99; // rax
  __m128i v100; // xmm0
  _QWORD *v101; // [rsp+8h] [rbp-128h]
  __int64 v102; // [rsp+10h] [rbp-120h]
  __int64 v103; // [rsp+18h] [rbp-118h]
  __int64 v104; // [rsp+30h] [rbp-100h]
  __int64 v105; // [rsp+38h] [rbp-F8h]
  _QWORD *v106; // [rsp+40h] [rbp-F0h]
  size_t v107; // [rsp+40h] [rbp-F0h]
  __int64 v108; // [rsp+48h] [rbp-E8h]
  __int64 v109; // [rsp+58h] [rbp-D8h] BYREF
  void *v110; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v111; // [rsp+68h] [rbp-C8h]
  _BYTE v112[48]; // [rsp+70h] [rbp-C0h] BYREF
  int v113; // [rsp+A0h] [rbp-90h]
  void *v114; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v115; // [rsp+B8h] [rbp-78h]
  _BYTE v116[48]; // [rsp+C0h] [rbp-70h] BYREF
  int v117; // [rsp+F0h] [rbp-40h]

  v2 = sub_BA8DC0(a2, (__int64)"llvm.mir.debugify", 17);
  if ( v2 )
  {
    v3 = *(__int64 **)(a1 + 8);
    v4 = v2;
    v5 = *v3;
    v6 = v3[1];
    if ( v5 == v6 )
LABEL_157:
      BUG();
    while ( *(_UNKNOWN **)v5 != &unk_50208C0 )
    {
      v5 += 16;
      if ( v6 == v5 )
        goto LABEL_157;
    }
    v102 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v5 + 8) + 104LL))(
             *(_QWORD *)(v5 + 8),
             &unk_50208C0)
         + 176;
    v7 = sub_B91A10(v4, 0);
    v8 = *(_BYTE *)(v7 - 16);
    if ( (v8 & 2) != 0 )
      v9 = *(_QWORD *)(v7 - 32);
    else
      v9 = v7 - 8LL * ((v8 >> 2) & 0xF) - 16;
    v10 = *(_QWORD *)(*(_QWORD *)v9 + 136LL);
    v11 = *(_QWORD **)(v10 + 24);
    if ( *(_DWORD *)(v10 + 32) > 0x40u )
      v11 = (_QWORD *)*v11;
    v12 = sub_B91A10(v4, 1u);
    v15 = *(_BYTE *)(v12 - 16);
    if ( (v15 & 2) != 0 )
      v16 = *(_QWORD *)(v12 - 32);
    else
      v16 = v12 - 8LL * ((v15 >> 2) & 0xF) - 16;
    v17 = *(_QWORD *)(*(_QWORD *)v16 + 136LL);
    v18 = *(_QWORD **)(v17 + 24);
    if ( *(_DWORD *)(v17 + 32) > 0x40u )
      v18 = (_QWORD *)*v18;
    v19 = (unsigned int)((_DWORD)v11 + 63) >> 6;
    v110 = v112;
    v111 = 0x600000000LL;
    if ( v19 > 6 )
    {
      sub_C8D5F0((__int64)&v110, v112, v19, 8u, v13, v14);
      memset(v110, 255, 8LL * v19);
      LODWORD(v111) = (unsigned int)((_DWORD)v11 + 63) >> 6;
    }
    else
    {
      if ( v19 && 8LL * v19 )
        memset(v112, 255, 8LL * v19);
      LODWORD(v111) = (unsigned int)((_DWORD)v11 + 63) >> 6;
    }
    v113 = (int)v11;
    if ( ((unsigned __int8)v11 & 0x3F) != 0 )
      *((_QWORD *)v110 + (unsigned int)v111 - 1) &= ~(-1LL << ((unsigned __int8)v11 & 0x3F));
    v20 = (unsigned int)((_DWORD)v18 + 63) >> 6;
    v114 = v116;
    v115 = 0x600000000LL;
    if ( v20 > 6 )
    {
      sub_C8D5F0((__int64)&v114, v116, v20, 8u, v13, v14);
      memset(v114, 255, 8LL * v20);
    }
    else if ( v20 && 8LL * v20 )
    {
      memset(v116, 255, 8LL * v20);
    }
    LODWORD(v115) = (unsigned int)((_DWORD)v18 + 63) >> 6;
    v117 = (int)v18;
    if ( ((unsigned __int8)v18 & 0x3F) != 0 )
      *((_QWORD *)v114 + (unsigned int)v115 - 1) &= ~(-1LL << ((unsigned __int8)v18 & 0x3F));
    v103 = a2 + 24;
    v104 = *(_QWORD *)(a2 + 32);
    if ( v104 != a2 + 24 )
    {
      while ( 1 )
      {
        v21 = 0;
        if ( v104 )
          v21 = v104 - 56;
        v105 = v21;
        v22 = sub_2EAA2D0(v102, v21);
        if ( v22 )
        {
          v23 = *(_QWORD *)(v22 + 328);
          v108 = v22 + 320;
          if ( v23 != v22 + 320 )
            break;
        }
LABEL_52:
        v104 = *(_QWORD *)(v104 + 8);
        if ( v103 == v104 )
          goto LABEL_53;
      }
LABEL_32:
      v24 = *(_QWORD *)(v23 + 56);
      v25 = v23 + 48;
      if ( v23 + 48 == v24 )
        goto LABEL_51;
      while ( (unsigned __int16)(*(_WORD *)(v24 + 68) - 14) <= 1u )
      {
LABEL_36:
        if ( (*(_BYTE *)v24 & 4) != 0 )
        {
          v24 = *(_QWORD *)(v24 + 8);
          if ( v25 == v24 )
            goto LABEL_38;
        }
        else
        {
          while ( (*(_BYTE *)(v24 + 44) & 8) != 0 )
            v24 = *(_QWORD *)(v24 + 8);
          v24 = *(_QWORD *)(v24 + 8);
          if ( v25 == v24 )
          {
LABEL_38:
            v26 = *(_QWORD *)(v23 + 56);
            if ( v26 == v24 )
            {
LABEL_51:
              v23 = *(_QWORD *)(v23 + 8);
              if ( v108 == v23 )
                goto LABEL_52;
              goto LABEL_32;
            }
            while ( 2 )
            {
              if ( (unsigned __int16)(*(_WORD *)(v26 + 68) - 14) <= 1u )
              {
                v27 = sub_2E89170(v26);
                v28 = *(_BYTE *)(v27 - 16);
                if ( (v28 & 2) != 0 )
                {
                  v29 = *(_QWORD *)(*(_QWORD *)(v27 - 32) + 8LL);
                  if ( v29 )
                    goto LABEL_44;
LABEL_93:
                  v31 = 0;
                }
                else
                {
                  v29 = *(_QWORD *)(v27 - 16 - 8LL * ((v28 >> 2) & 0xF) + 8);
                  if ( !v29 )
                    goto LABEL_93;
LABEL_44:
                  v29 = sub_B91420(v29);
                  v31 = v30;
                }
                if ( sub_C93C90(v29, v31, 0xAu, (unsigned __int64 *)&v109) || v109 != (unsigned int)v109 )
                {
                  v32 = 0xBFFFFFFFFFFFFFFFLL;
                  v33 = 536870904;
                }
                else
                {
                  v32 = ~(1LL << ((unsigned __int8)v109 - 1));
                  v33 = 8LL * ((unsigned int)(v109 - 1) >> 6);
                }
                *(_QWORD *)((char *)v114 + v33) &= v32;
              }
              if ( (*(_BYTE *)v26 & 4) != 0 )
              {
                v26 = *(_QWORD *)(v26 + 8);
                if ( v26 == v24 )
                  goto LABEL_51;
              }
              else
              {
                while ( (*(_BYTE *)(v26 + 44) & 8) != 0 )
                  v26 = *(_QWORD *)(v26 + 8);
                v26 = *(_QWORD *)(v26 + 8);
                if ( v26 == v24 )
                  goto LABEL_51;
              }
              continue;
            }
          }
        }
      }
      v49 = *(_QWORD *)(v24 + 56);
      v109 = v49;
      if ( !v49 )
        goto LABEL_94;
      sub_B96E90((__int64)&v109, v49, 1);
      if ( !v109 )
        goto LABEL_94;
      if ( (unsigned int)sub_B10CE0((__int64)&v109) )
      {
        v61 = sub_B10CE0((__int64)&v109);
        *((_QWORD *)v110 + ((unsigned int)(v61 - 1) >> 6)) &= ~(1LL << ((unsigned __int8)v61 - 1));
        v50 = v109;
        if ( !v109 )
          goto LABEL_36;
        goto LABEL_87;
      }
      v50 = v109;
      if ( !v109 )
      {
LABEL_94:
        v51 = sub_CB72A0();
        v52 = (__m128i *)v51[4];
        if ( v51[3] - (_QWORD)v52 <= 0x34u )
        {
          sub_CB6200((__int64)v51, "WARNING: Instruction with empty DebugLoc in function ", 0x35u);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_44E4D10);
          v52[3].m128i_i32[0] = 1852795252;
          v52[3].m128i_i8[4] = 32;
          *v52 = si128;
          v52[1] = _mm_load_si128((const __m128i *)&xmmword_44E4D20);
          v52[2] = _mm_load_si128((const __m128i *)&xmmword_44E4D30);
          v51[4] += 53LL;
        }
        v106 = sub_CB72A0();
        v54 = sub_BD5D20(v105);
        v56 = (__int64)v106;
        v57 = (unsigned __int8 *)v54;
        v58 = (_BYTE *)v106[4];
        v59 = v106[3] - (_QWORD)v58;
        if ( v59 < v55 )
        {
          v62 = sub_CB6200((__int64)v106, v57, v55);
          v58 = *(_BYTE **)(v62 + 32);
          v56 = v62;
          v59 = *(_QWORD *)(v62 + 24) - (_QWORD)v58;
        }
        else if ( v55 )
        {
          v101 = v106;
          v107 = v55;
          memcpy(v58, v57, v55);
          v56 = (__int64)v101;
          v63 = v101[3];
          v58 = (_BYTE *)(v101[4] + v107);
          v101[4] = v58;
          v59 = v63 - (_QWORD)v58;
        }
        if ( v59 <= 2 )
        {
          sub_CB6200(v56, " --", 3u);
        }
        else
        {
          v58[2] = 45;
          *(_WORD *)v58 = 11552;
          *(_QWORD *)(v56 + 32) += 3LL;
        }
        v60 = sub_CB72A0();
        sub_2E91850(v24, (__int64)v60, 1u, 0, 0, 1, 0);
        v50 = v109;
        if ( !v109 )
          goto LABEL_36;
      }
LABEL_87:
      sub_B91220((__int64)&v109, v50);
      goto LABEL_36;
    }
LABEL_53:
    if ( !v113 )
      goto LABEL_59;
    v34 = (unsigned int)(v113 - 1) >> 6;
    v35 = 0;
    while ( 1 )
    {
      _RDX = *((_QWORD *)v110 + v35);
      if ( v34 == (_DWORD)v35 )
        _RDX = (0xFFFFFFFFFFFFFFFFLL >> -(char)v113) & *((_QWORD *)v110 + v35);
      if ( _RDX )
        break;
      if ( v34 + 1 == ++v35 )
        goto LABEL_59;
    }
    __asm { tzcnt   r12, rdx }
    v82 = ((_DWORD)v35 << 6) + _R12;
    if ( v82 == -1 )
    {
LABEL_59:
      v37 = 0;
    }
    else
    {
      do
      {
        v83 = sub_CB72A0();
        v84 = (__m128i *)v83[4];
        v85 = (__int64)v83;
        if ( v83[3] - (_QWORD)v84 <= 0x15u )
        {
          v85 = sub_CB6200((__int64)v83, "WARNING: Missing line ", 0x16u);
        }
        else
        {
          v86 = _mm_load_si128((const __m128i *)&xmmword_439AB30);
          v84[1].m128i_i32[0] = 1852402720;
          v84[1].m128i_i16[2] = 8293;
          *v84 = v86;
          v83[4] += 22LL;
        }
        v87 = v82 + 1;
        v88 = sub_CB59D0(v85, v87);
        v89 = *(_BYTE **)(v88 + 32);
        if ( *(_BYTE **)(v88 + 24) == v89 )
        {
          sub_CB6200(v88, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *v89 = 10;
          ++*(_QWORD *)(v88 + 32);
        }
        if ( v87 == v113 )
          break;
        v90 = v87 >> 6;
        v91 = (unsigned int)(v113 - 1) >> 6;
        if ( v87 >> 6 > v91 )
          break;
        v92 = v87 & 0x3F;
        v93 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v92);
        if ( v92 == 0 )
          v93 = 0;
        v94 = v90;
        v95 = ~v93;
        while ( 1 )
        {
          _RAX = *((_QWORD *)v110 + v94);
          if ( v90 == (_DWORD)v94 )
            _RAX = v95 & *((_QWORD *)v110 + v94);
          if ( v91 == (_DWORD)v94 )
            _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v113;
          if ( _RAX )
            break;
          if ( v91 < (unsigned int)++v94 )
            goto LABEL_145;
        }
        __asm { tzcnt   rax, rax }
        v82 = ((_DWORD)v94 << 6) + _RAX;
      }
      while ( v82 != -1 );
LABEL_145:
      v37 = 1;
    }
    if ( v117 )
    {
      v38 = (unsigned int)(v117 - 1) >> 6;
      v39 = 0;
      while ( 1 )
      {
        _RDX = *((_QWORD *)v114 + v39);
        if ( v38 == (_DWORD)v39 )
          _RDX = (0xFFFFFFFFFFFFFFFFLL >> -(char)v117) & *((_QWORD *)v114 + v39);
        if ( _RDX )
          break;
        if ( v38 + 1 == ++v39 )
          goto LABEL_66;
      }
      __asm { tzcnt   r12, rdx }
      v65 = ((_DWORD)v39 << 6) + _R12;
      if ( v65 != -1 )
      {
        do
        {
          v66 = sub_CB72A0();
          v67 = (__m128i *)v66[4];
          v68 = (__int64)v66;
          if ( v66[3] - (_QWORD)v67 <= 0x19u )
          {
            v68 = sub_CB6200((__int64)v66, "WARNING: Missing variable ", 0x1Au);
          }
          else
          {
            v69 = _mm_load_si128((const __m128i *)&xmmword_439AB30);
            qmemcpy(&v67[1], " variable ", 10);
            *v67 = v69;
            v66[4] += 26LL;
          }
          v70 = v65 + 1;
          v71 = sub_CB59D0(v68, v70);
          v72 = *(_BYTE **)(v71 + 32);
          if ( *(_BYTE **)(v71 + 24) == v72 )
          {
            sub_CB6200(v71, (unsigned __int8 *)"\n", 1u);
          }
          else
          {
            *v72 = 10;
            ++*(_QWORD *)(v71 + 32);
          }
          if ( v70 == v117 )
            break;
          v73 = v70 >> 6;
          v74 = (unsigned int)(v117 - 1) >> 6;
          if ( v70 >> 6 > v74 )
            break;
          v75 = v70 & 0x3F;
          v76 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v75);
          if ( v75 == 0 )
            v76 = 0;
          v77 = v73;
          v78 = ~v76;
          while ( 1 )
          {
            _RAX = *((_QWORD *)v114 + v77);
            if ( v73 == (_DWORD)v77 )
              _RAX = v78 & *((_QWORD *)v114 + v77);
            if ( v74 == (_DWORD)v77 )
              _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v117;
            if ( _RAX )
              break;
            if ( v74 < (unsigned int)++v77 )
              goto LABEL_125;
          }
          __asm { tzcnt   rax, rax }
          v65 = ((_DWORD)v77 << 6) + _RAX;
        }
        while ( v65 != -1 );
LABEL_125:
        v37 = 1;
      }
    }
LABEL_66:
    v41 = sub_CB72A0();
    v42 = (__m128i *)v41[4];
    if ( v41[3] - (_QWORD)v42 <= 0x1Cu )
    {
      sub_CB6200((__int64)v41, "Machine IR debug info check: ", 0x1Du);
    }
    else
    {
      v43 = _mm_load_si128((const __m128i *)&xmmword_44E4D40);
      qmemcpy(&v42[1], " info check: ", 13);
      *v42 = v43;
      v41[4] += 29LL;
    }
    v44 = "FAIL";
    v45 = (__int64)sub_CB72A0();
    if ( !v37 )
      v44 = "PASS";
    v46 = *(_DWORD **)(v45 + 32);
    if ( *(_QWORD *)(v45 + 24) - (_QWORD)v46 > 3u )
    {
      *v46 = *(_DWORD *)v44;
      v47 = (_BYTE *)(*(_QWORD *)(v45 + 32) + 4LL);
      *(_QWORD *)(v45 + 32) = v47;
    }
    else
    {
      v45 = sub_CB6200(v45, (unsigned __int8 *)v44, 4u);
      v47 = *(_BYTE **)(v45 + 32);
    }
    if ( *(_BYTE **)(v45 + 24) == v47 )
    {
      sub_CB6200(v45, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v47 = 10;
      ++*(_QWORD *)(v45 + 32);
    }
    if ( v114 != v116 )
      _libc_free((unsigned __int64)v114);
    if ( v110 != v112 )
      _libc_free((unsigned __int64)v110);
  }
  else
  {
    v98 = sub_CB72A0();
    v99 = (__m128i *)v98[4];
    if ( v98[3] - (_QWORD)v99 <= 0x4Eu )
    {
      sub_CB6200(
        (__int64)v98,
        "WARNING: Please run mir-debugify to generate llvm.mir.debugify metadata first.\n",
        0x4Fu);
    }
    else
    {
      v100 = _mm_load_si128((const __m128i *)&xmmword_44E4CD0);
      qmemcpy(&v99[4], "etadata first.\n", 15);
      *v99 = v100;
      v99[1] = _mm_load_si128((const __m128i *)&xmmword_44E4CE0);
      v99[2] = _mm_load_si128((const __m128i *)&xmmword_44E4CF0);
      v99[3] = _mm_load_si128((const __m128i *)&xmmword_44E4D00);
      v98[4] += 79LL;
    }
  }
  return 0;
}
