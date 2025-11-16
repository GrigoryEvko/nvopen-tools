// Function: sub_2FCD340
// Address: 0x2fcd340
//
__int64 __fastcall sub_2FCD340(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 *v3; // rax
  unsigned int v4; // r14d
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __m128i v13; // xmm0
  __m128i v14; // xmm2
  unsigned __int64 *v15; // rbx
  unsigned __int64 *v16; // r15
  unsigned __int64 v17; // rdi
  unsigned __int64 *v18; // rbx
  unsigned __int64 *v19; // r12
  unsigned __int64 v20; // rdi
  __int64 v21; // rbx
  unsigned __int8 v22; // r12
  __int64 v23; // rdx
  unsigned __int64 v24; // rcx
  unsigned __int64 v25; // rax
  __int64 *v26; // r12
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // r15
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __m128i v34; // xmm3
  __m128i v35; // xmm5
  unsigned __int64 *v36; // r15
  __int64 v37; // r8
  unsigned __int64 *v38; // r14
  unsigned __int64 v39; // rdi
  unsigned __int64 *v40; // r15
  unsigned __int64 *v41; // r14
  unsigned __int64 v42; // rdi
  char v43; // r12
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // r12
  __int64 v47; // rax
  __int64 v48; // r12
  __int64 v49; // rdx
  __int64 v50; // r8
  __int64 v51; // r9
  __m128i v52; // xmm6
  __m128i v53; // xmm6
  __int64 v54; // rcx
  unsigned __int64 *v55; // r12
  unsigned __int64 *v56; // r15
  unsigned __int64 v57; // rdi
  unsigned __int64 *v58; // r12
  unsigned __int64 *v59; // r14
  unsigned __int64 v60; // rdi
  unsigned int v61; // eax
  __int64 *v62; // rax
  __int64 *v63; // rdx
  unsigned int v64; // eax
  unsigned int v65; // r12d
  char v66; // al
  __int64 v67; // rdi
  __int64 v68; // rax
  __int64 *v69; // rdx
  __int64 *v70; // rax
  __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 *v77; // rax
  __int64 *v78; // rdx
  unsigned __int8 v79; // [rsp+8h] [rbp-698h]
  __int64 v80; // [rsp+10h] [rbp-690h]
  __int64 v81; // [rsp+10h] [rbp-690h]
  unsigned __int8 *v82; // [rsp+18h] [rbp-688h]
  int v83; // [rsp+20h] [rbp-680h]
  __int64 v84; // [rsp+28h] [rbp-678h]
  unsigned __int8 v85; // [rsp+30h] [rbp-670h]
  __int64 v86; // [rsp+30h] [rbp-670h]
  unsigned __int8 v87; // [rsp+47h] [rbp-659h]
  __int64 v88; // [rsp+48h] [rbp-658h]
  unsigned int v89; // [rsp+58h] [rbp-648h]
  unsigned __int8 *v90; // [rsp+68h] [rbp-638h] BYREF
  char v91; // [rsp+7Fh] [rbp-621h] BYREF
  unsigned __int64 v92; // [rsp+80h] [rbp-620h]
  __int64 v93; // [rsp+88h] [rbp-618h]
  __int64 v94[2]; // [rsp+90h] [rbp-610h] BYREF
  __int64 *v95; // [rsp+A0h] [rbp-600h]
  unsigned __int64 v96[2]; // [rsp+B0h] [rbp-5F0h] BYREF
  _QWORD v97[2]; // [rsp+C0h] [rbp-5E0h] BYREF
  _QWORD *v98; // [rsp+D0h] [rbp-5D0h]
  _QWORD v99[4]; // [rsp+E0h] [rbp-5C0h] BYREF
  void *v100; // [rsp+100h] [rbp-5A0h] BYREF
  int v101; // [rsp+108h] [rbp-598h] BYREF
  char v102; // [rsp+10Ch] [rbp-594h]
  __int64 v103; // [rsp+110h] [rbp-590h]
  __m128i v104; // [rsp+118h] [rbp-588h]
  __int64 v105; // [rsp+128h] [rbp-578h]
  __m128i v106; // [rsp+130h] [rbp-570h]
  __m128i v107; // [rsp+140h] [rbp-560h]
  unsigned __int64 *v108; // [rsp+150h] [rbp-550h] BYREF
  __int64 v109; // [rsp+158h] [rbp-548h]
  _BYTE v110[320]; // [rsp+160h] [rbp-540h] BYREF
  char v111; // [rsp+2A0h] [rbp-400h]
  int v112; // [rsp+2A4h] [rbp-3FCh]
  __int64 v113; // [rsp+2A8h] [rbp-3F8h]
  _QWORD v114[10]; // [rsp+2B0h] [rbp-3F0h] BYREF
  unsigned __int64 *v115; // [rsp+300h] [rbp-3A0h]
  unsigned int v116; // [rsp+308h] [rbp-398h]
  _BYTE v117[336]; // [rsp+310h] [rbp-390h] BYREF
  __int64 v118; // [rsp+460h] [rbp-240h] BYREF
  __int64 v119; // [rsp+468h] [rbp-238h]
  __int64 *v120; // [rsp+470h] [rbp-230h] BYREF
  unsigned int v121; // [rsp+478h] [rbp-228h]
  _BYTE v122[48]; // [rsp+670h] [rbp-30h] BYREF

  v90 = (unsigned __int8 *)a1;
  v2 = *(_QWORD *)(a1 + 40);
  v118 = 0;
  v119 = 1;
  v88 = v2;
  v3 = (__int64 *)&v120;
  do
  {
    *v3 = -4096;
    v3 += 4;
  }
  while ( v3 != (__int64 *)v122 );
  v4 = 0;
  v89 = sub_B2D810(a1, "stack-protector-buffer-size", 0x1Bu, 8);
  if ( (unsigned __int8)sub_B2D610((__int64)v90, 55) )
    goto LABEL_4;
  sub_1049690(v94, (__int64)v90);
  v4 = sub_B2D610((__int64)v90, 70);
  if ( (_BYTE)v4 )
  {
    if ( !a2 )
      goto LABEL_53;
    v6 = v94[0];
    v7 = sub_B2BE50(v94[0]);
    if ( sub_B6EA50(v7)
      || (v71 = sub_B2BE50(v6),
          v72 = sub_B6F970(v71),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v72 + 48LL))(v72)) )
    {
      sub_B17560((__int64)v114, (__int64)"stack-protector", (__int64)"StackProtectorRequested", 23, (__int64)v90);
      sub_B18290((__int64)v114, "Stack protection applied to function ", 0x25u);
      sub_B16080((__int64)v96, "Function", 8, v90);
      v8 = sub_23FD640((__int64)v114, (__int64)v96);
      sub_B18290(v8, " due to a function attribute or command-line switch", 0x33u);
      v101 = *(_DWORD *)(v8 + 8);
      v102 = *(_BYTE *)(v8 + 12);
      v103 = *(_QWORD *)(v8 + 16);
      v13 = _mm_loadu_si128((const __m128i *)(v8 + 24));
      v100 = &unk_49D9D40;
      v104 = v13;
      v105 = *(_QWORD *)(v8 + 40);
      v106 = _mm_loadu_si128((const __m128i *)(v8 + 48));
      v14 = _mm_loadu_si128((const __m128i *)(v8 + 64));
      v108 = (unsigned __int64 *)v110;
      v109 = 0x400000000LL;
      v107 = v14;
      if ( *(_DWORD *)(v8 + 88) )
        sub_2FCBCE0((__int64)&v108, v8 + 80, v9, v10, v11, v12);
      v111 = *(_BYTE *)(v8 + 416);
      v112 = *(_DWORD *)(v8 + 420);
      v113 = *(_QWORD *)(v8 + 424);
      v100 = &unk_49D9D78;
      if ( v98 != v99 )
        j_j___libc_free_0((unsigned __int64)v98);
      if ( (_QWORD *)v96[0] != v97 )
        j_j___libc_free_0(v96[0]);
      v15 = v115;
      v114[0] = &unk_49D9D40;
      v16 = &v115[10 * v116];
      if ( v115 != v16 )
      {
        do
        {
          v16 -= 10;
          v17 = v16[4];
          if ( (unsigned __int64 *)v17 != v16 + 6 )
            j_j___libc_free_0(v17);
          if ( (unsigned __int64 *)*v16 != v16 + 2 )
            j_j___libc_free_0(*v16);
        }
        while ( v15 != v16 );
        v16 = v115;
      }
      if ( v16 != (unsigned __int64 *)v117 )
        _libc_free((unsigned __int64)v16);
      sub_1049740(v94, (__int64)&v100);
      v18 = v108;
      v100 = &unk_49D9D40;
      v19 = &v108[10 * (unsigned int)v109];
      if ( v108 != v19 )
      {
        do
        {
          v19 -= 10;
          v20 = v19[4];
          if ( (unsigned __int64 *)v20 != v19 + 6 )
            j_j___libc_free_0(v20);
          if ( (unsigned __int64 *)*v19 != v19 + 2 )
            j_j___libc_free_0(*v19);
        }
        while ( v18 != v19 );
        v19 = v108;
      }
      if ( v19 != (unsigned __int64 *)v110 )
        _libc_free((unsigned __int64)v19);
    }
    v87 = v4;
  }
  else
  {
    v87 = sub_B2D610((__int64)v90, 71);
    if ( !v87 )
    {
      v4 = sub_B2D610((__int64)v90, 69);
      if ( !(_BYTE)v4 )
        goto LABEL_53;
      v4 = 0;
    }
  }
  v82 = v90 + 72;
  v84 = *((_QWORD *)v90 + 10);
  if ( (unsigned __int8 *)v84 != v90 + 72 )
  {
    v85 = v4;
    while ( 1 )
    {
      if ( !v84 )
        BUG();
      if ( v84 + 24 != *(_QWORD *)(v84 + 32) )
        break;
LABEL_51:
      v84 = *(_QWORD *)(v84 + 8);
      if ( v82 == (unsigned __int8 *)v84 )
      {
        v4 = v85;
        goto LABEL_53;
      }
    }
    v21 = *(_QWORD *)(v84 + 32);
    while ( 1 )
    {
      if ( !v21 )
        BUG();
      if ( *(_BYTE *)(v21 - 24) != 60 )
        goto LABEL_50;
      v22 = sub_B4CE70(v21 - 24);
      if ( v22 )
      {
        v23 = *(_QWORD *)(v21 - 56);
        if ( *(_BYTE *)v23 != 17 )
          goto LABEL_46;
        v24 = v89;
        if ( *(_DWORD *)(v23 + 32) > 0x40u )
        {
          v80 = *(_QWORD *)(v21 - 56);
          v83 = *(_DWORD *)(v23 + 32);
          if ( v83 - (unsigned int)sub_C444A0(v23 + 24) <= 0x40 )
          {
            v24 = v89;
            v25 = **(_QWORD **)(v80 + 24);
            if ( v89 >= v25 )
              goto LABEL_85;
          }
        }
        else
        {
          v25 = *(_QWORD *)(v23 + 24);
          if ( v89 < v25 )
            goto LABEL_46;
LABEL_85:
          if ( v24 > v25 )
          {
            v22 = v87;
            if ( !v87 )
              goto LABEL_50;
            if ( !a2 )
              goto LABEL_55;
            v100 = (void *)(v21 - 24);
            v101 = 2;
            goto LABEL_48;
          }
        }
LABEL_46:
        if ( !a2 )
          goto LABEL_55;
        v100 = (void *)(v21 - 24);
        v101 = 1;
LABEL_48:
        sub_2FCD090((__int64)v114, a2, (__int64 *)&v100, &v101);
        sub_2FCBF60(v94, v21 - 24, &v90);
LABEL_49:
        v85 = v22;
        goto LABEL_50;
      }
      v27 = *(_QWORD *)(v21 + 48);
      v91 = 0;
      v22 = sub_2FC9AB0(v27, v88, v89, &v91, v87, 0);
      if ( v22 )
      {
        if ( !a2 )
          goto LABEL_55;
        v100 = (void *)(v21 - 24);
        v101 = (v91 == 0) + 1;
        sub_2FCD090((__int64)v114, a2, (__int64 *)&v100, &v101);
        v86 = v94[0];
        v28 = sub_B2BE50(v94[0]);
        if ( sub_B6EA50(v28)
          || (v73 = sub_B2BE50(v86),
              v74 = sub_B6F970(v73),
              (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v74 + 48LL))(v74)) )
        {
          sub_B174A0((__int64)v114, (__int64)"stack-protector", (__int64)"StackProtectorBuffer", 20, v21 - 24);
          sub_B18290((__int64)v114, "Stack protection applied to function ", 0x25u);
          sub_B16080((__int64)v96, "Function", 8, v90);
          v29 = sub_23FD640((__int64)v114, (__int64)v96);
          sub_B18290(v29, " due to a stack allocated buffer or struct containing a buffer", 0x3Eu);
          v101 = *(_DWORD *)(v29 + 8);
          v102 = *(_BYTE *)(v29 + 12);
          v103 = *(_QWORD *)(v29 + 16);
          v34 = _mm_loadu_si128((const __m128i *)(v29 + 24));
          v100 = &unk_49D9D40;
          v104 = v34;
          v105 = *(_QWORD *)(v29 + 40);
          v106 = _mm_loadu_si128((const __m128i *)(v29 + 48));
          v35 = _mm_loadu_si128((const __m128i *)(v29 + 64));
          v108 = (unsigned __int64 *)v110;
          v109 = 0x400000000LL;
          v107 = v35;
          if ( *(_DWORD *)(v29 + 88) )
            sub_2FCBCE0((__int64)&v108, v29 + 80, v30, v31, v32, v33);
          v111 = *(_BYTE *)(v29 + 416);
          v112 = *(_DWORD *)(v29 + 420);
          v113 = *(_QWORD *)(v29 + 424);
          v100 = &unk_49D9D78;
          if ( v98 != v99 )
            j_j___libc_free_0((unsigned __int64)v98);
          if ( (_QWORD *)v96[0] != v97 )
            j_j___libc_free_0(v96[0]);
          v36 = v115;
          v114[0] = &unk_49D9D40;
          v37 = 10LL * v116;
          v38 = &v115[v37];
          if ( v115 != &v115[v37] )
          {
            do
            {
              v38 -= 10;
              v39 = v38[4];
              if ( (unsigned __int64 *)v39 != v38 + 6 )
                j_j___libc_free_0(v39);
              if ( (unsigned __int64 *)*v38 != v38 + 2 )
                j_j___libc_free_0(*v38);
            }
            while ( v36 != v38 );
            v38 = v115;
          }
          if ( v38 != (unsigned __int64 *)v117 )
            _libc_free((unsigned __int64)v38);
          sub_1049740(v94, (__int64)&v100);
          v40 = v108;
          v100 = &unk_49D9D40;
          v41 = &v108[10 * (unsigned int)v109];
          if ( v108 != v41 )
          {
            do
            {
              v41 -= 10;
              v42 = v41[4];
              if ( (unsigned __int64 *)v42 != v41 + 6 )
                j_j___libc_free_0(v42);
              if ( (unsigned __int64 *)*v41 != v41 + 2 )
                j_j___libc_free_0(*v41);
            }
            while ( v40 != v41 );
            v41 = v108;
          }
          if ( v41 != (unsigned __int64 *)v110 )
            _libc_free((unsigned __int64)v41);
        }
        goto LABEL_49;
      }
      if ( v87 )
      {
        v81 = *(_QWORD *)(v21 + 48);
        v43 = sub_AE5020(v88 + 312, v81);
        v44 = sub_9208B0(v88 + 312, v81);
        v114[1] = v45;
        v114[0] = v44;
        LOBYTE(v93) = v45;
        v92 = ((1LL << v43) + ((unsigned __int64)(v44 + 7) >> 3) - 1) >> v43 << v43;
        v79 = sub_2FCCAF0(v21 - 24, v92, v93, v88, (__int64)&v118);
        if ( v79 )
        {
          if ( !a2 )
          {
LABEL_55:
            v4 = 1;
            goto LABEL_53;
          }
          v100 = (void *)(v21 - 24);
          v101 = 3;
          sub_2FCD090((__int64)v114, a2, (__int64 *)&v100, &v101);
          v46 = v94[0];
          v47 = sub_B2BE50(v94[0]);
          if ( sub_B6EA50(v47)
            || (v75 = sub_B2BE50(v46),
                v76 = sub_B6F970(v75),
                (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v76 + 48LL))(v76)) )
          {
            sub_B174A0((__int64)v114, (__int64)"stack-protector", (__int64)"StackProtectorAddressTaken", 26, v21 - 24);
            sub_B18290((__int64)v114, "Stack protection applied to function ", 0x25u);
            sub_B16080((__int64)v96, "Function", 8, v90);
            v48 = sub_23FD640((__int64)v114, (__int64)v96);
            sub_B18290(v48, " due to the address of a local variable being taken", 0x33u);
            v101 = *(_DWORD *)(v48 + 8);
            v102 = *(_BYTE *)(v48 + 12);
            v103 = *(_QWORD *)(v48 + 16);
            v52 = _mm_loadu_si128((const __m128i *)(v48 + 24));
            v100 = &unk_49D9D40;
            v104 = v52;
            v105 = *(_QWORD *)(v48 + 40);
            v106 = _mm_loadu_si128((const __m128i *)(v48 + 48));
            v53 = _mm_loadu_si128((const __m128i *)(v48 + 64));
            v108 = (unsigned __int64 *)v110;
            v109 = 0x400000000LL;
            v107 = v53;
            v54 = *(unsigned int *)(v48 + 88);
            if ( (_DWORD)v54 )
              sub_2FCBCE0((__int64)&v108, v48 + 80, v49, v54, v50, v51);
            v111 = *(_BYTE *)(v48 + 416);
            v112 = *(_DWORD *)(v48 + 420);
            v113 = *(_QWORD *)(v48 + 424);
            v100 = &unk_49D9D78;
            if ( v98 != v99 )
              j_j___libc_free_0((unsigned __int64)v98);
            if ( (_QWORD *)v96[0] != v97 )
              j_j___libc_free_0(v96[0]);
            v55 = v115;
            v114[0] = &unk_49D9D40;
            v56 = &v115[10 * v116];
            if ( v115 != v56 )
            {
              do
              {
                v56 -= 10;
                v57 = v56[4];
                if ( (unsigned __int64 *)v57 != v56 + 6 )
                  j_j___libc_free_0(v57);
                if ( (unsigned __int64 *)*v56 != v56 + 2 )
                  j_j___libc_free_0(*v56);
              }
              while ( v55 != v56 );
              v56 = v115;
            }
            if ( v56 != (unsigned __int64 *)v117 )
              _libc_free((unsigned __int64)v56);
            sub_1049740(v94, (__int64)&v100);
            v58 = v108;
            v100 = &unk_49D9D40;
            v59 = &v108[10 * (unsigned int)v109];
            if ( v108 != v59 )
            {
              do
              {
                v59 -= 10;
                v60 = v59[4];
                if ( (unsigned __int64 *)v60 != v59 + 6 )
                  j_j___libc_free_0(v60);
                if ( (unsigned __int64 *)*v59 != v59 + 2 )
                  j_j___libc_free_0(*v59);
              }
              while ( v58 != v59 );
              v59 = v108;
            }
            if ( v59 != (unsigned __int64 *)v110 )
              _libc_free((unsigned __int64)v59);
          }
          v85 = v79;
        }
      }
      ++v118;
      v61 = (unsigned int)v119 >> 1;
      if ( (unsigned int)v119 >> 1 || HIDWORD(v119) )
        break;
LABEL_50:
      v21 = *(_QWORD *)(v21 + 8);
      if ( v84 + 24 == v21 )
        goto LABEL_51;
    }
    if ( (v119 & 1) != 0 )
    {
      v63 = (__int64 *)v122;
      v62 = (__int64 *)&v120;
      goto LABEL_123;
    }
    if ( 4 * v61 >= v121 || v121 <= 0x40 )
    {
      v62 = v120;
      v63 = &v120[4 * v121];
      if ( v120 != v63 )
      {
        do
        {
LABEL_123:
          *v62 = -4096;
          v62 += 4;
        }
        while ( v63 != v62 );
      }
      v119 &= 1u;
      goto LABEL_50;
    }
    if ( v61 && (v64 = v61 - 1) != 0 )
    {
      _BitScanReverse(&v64, v64);
      v65 = 1 << (33 - (v64 ^ 0x1F));
      if ( v65 - 17 > 0x2E )
      {
        if ( v121 == v65 )
        {
          v119 &= 1u;
          if ( v119 )
          {
            v78 = (__int64 *)v122;
            v77 = (__int64 *)&v120;
          }
          else
          {
            v77 = v120;
            v78 = &v120[4 * v121];
          }
          do
          {
            if ( v77 )
              *v77 = -4096;
            v77 += 4;
          }
          while ( v78 != v77 );
          goto LABEL_50;
        }
        sub_C7D6A0((__int64)v120, 32LL * v121, 8);
        v66 = v119 | 1;
        LOBYTE(v119) = v119 | 1;
        if ( v65 <= 0x10 )
          goto LABEL_135;
        v67 = 32LL * v65;
      }
      else
      {
        v65 = 64;
        sub_C7D6A0((__int64)v120, 32LL * v121, 8);
        v66 = v119;
        v67 = 2048;
      }
      LOBYTE(v119) = v66 & 0xFE;
      v68 = sub_C7D670(v67, 8);
      v121 = v65;
      v120 = (__int64 *)v68;
    }
    else
    {
      sub_C7D6A0((__int64)v120, 32LL * v121, 8);
      LOBYTE(v119) = v119 | 1;
    }
LABEL_135:
    v119 &= 1u;
    if ( v119 )
    {
      v69 = (__int64 *)v122;
      v70 = (__int64 *)&v120;
      do
      {
LABEL_137:
        if ( v70 )
          *v70 = -4096;
        v70 += 4;
      }
      while ( v69 != v70 );
      goto LABEL_50;
    }
    v70 = v120;
    v69 = &v120[4 * v121];
    if ( v120 != v69 )
      goto LABEL_137;
    goto LABEL_50;
  }
LABEL_53:
  v26 = v95;
  if ( v95 )
  {
    sub_FDC110(v95);
    j_j___libc_free_0((unsigned __int64)v26);
  }
LABEL_4:
  if ( (v119 & 1) == 0 )
    sub_C7D6A0((__int64)v120, 32LL * v121, 8);
  return v4;
}
