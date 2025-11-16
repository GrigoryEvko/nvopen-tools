// Function: sub_20D8B60
// Address: 0x20d8b60
//
__int64 __fastcall sub_20D8B60(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  bool v3; // r12
  __int64 *v4; // rsi
  __int64 v5; // rdi
  int v6; // r8d
  unsigned int v7; // ecx
  __int64 *v8; // rax
  __int64 v9; // rdx
  unsigned int v10; // ecx
  __int64 *v11; // rdx
  __int64 v12; // r10
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // rdi
  __int64 (*v16)(); // rax
  __int64 v17; // rdi
  __int64 (*v18)(); // rax
  __int64 *v19; // r13
  __int64 *v20; // r14
  __int64 v21; // r15
  __int64 v22; // rdi
  __int64 (*v23)(); // rax
  __int64 v24; // r14
  __int64 v25; // r15
  _QWORD **v26; // r14
  _QWORD **v27; // r12
  _QWORD *v28; // r13
  __int64 *v29; // rdi
  unsigned __int64 v30; // rdi
  __int64 v31; // rdi
  __int64 (*v32)(); // rax
  bool v33; // r13
  __int64 *v35; // r14
  __int64 v36; // r13
  __int64 v37; // r15
  __int64 v38; // rax
  __int64 v39; // r12
  __int64 v40; // rbx
  __int64 v41; // rcx
  int v42; // r8d
  __int64 v43; // rdx
  __int64 v44; // rdx
  int v45; // r8d
  int v46; // r9d
  __int64 v47; // rdi
  __int64 (*v48)(); // rax
  __int64 v49; // rcx
  int v50; // r8d
  int v51; // r9d
  __int64 v52; // rdi
  __int16 v53; // ax
  unsigned __int8 v54; // r15
  unsigned __int64 v55; // r14
  __int64 v56; // r13
  __int64 v57; // rdx
  __int64 v58; // rax
  __int64 v59; // r12
  __int64 v60; // rdi
  __int64 v61; // rdi
  __int64 (*v62)(); // rax
  __int64 v63; // r12
  __int64 v64; // r15
  __int64 (*v65)(); // rax
  __int64 v66; // r11
  __int64 (*v67)(); // rax
  __int64 v68; // rdi
  __int64 (*v69)(); // rax
  int v70; // edx
  int v71; // eax
  __int64 *v72; // r14
  __int64 v73; // r15
  __int64 v74; // r13
  unsigned __int64 v75; // rax
  __int64 v76; // r12
  unsigned __int64 v77; // rbx
  bool v78; // r12
  __int64 j; // rax
  __int64 v80; // rdi
  int v81; // r9d
  int v82; // r9d
  __int64 v83; // rdi
  __int64 (*v84)(); // rax
  unsigned __int8 v85; // al
  unsigned __int8 v86; // r14
  unsigned __int64 v87; // r12
  unsigned __int64 v88; // rax
  __int64 v89; // r15
  __int16 v90; // ax
  __int16 v91; // ax
  __int64 v92; // rax
  __int64 v93; // rcx
  __int64 v94; // rdx
  unsigned __int8 v95; // r14
  unsigned __int8 v96; // r12
  __int64 v97; // rdi
  __int64 v98; // r15
  __int64 v99; // rdi
  __int64 v100; // rdi
  __int16 v101; // cx
  unsigned __int64 *v102; // r12
  unsigned __int64 v103; // rdx
  __int64 v104; // rax
  _QWORD *v105; // r13
  __int64 v106; // rbx
  _QWORD *v107; // r12
  unsigned __int64 *v108; // rcx
  unsigned __int64 v109; // rdx
  unsigned __int8 v110; // [rsp+10h] [rbp-2D0h]
  __int64 v111; // [rsp+28h] [rbp-2B8h]
  _QWORD *v112; // [rsp+30h] [rbp-2B0h]
  __int64 v113; // [rsp+38h] [rbp-2A8h]
  unsigned int v114; // [rsp+40h] [rbp-2A0h]
  char v115; // [rsp+46h] [rbp-29Ah]
  unsigned __int8 v116; // [rsp+47h] [rbp-299h]
  unsigned __int64 v117; // [rsp+48h] [rbp-298h]
  bool v118; // [rsp+50h] [rbp-290h]
  __int64 v119; // [rsp+50h] [rbp-290h]
  __int64 v120; // [rsp+50h] [rbp-290h]
  __int64 v121; // [rsp+50h] [rbp-290h]
  __int64 v122; // [rsp+50h] [rbp-290h]
  __int64 v123; // [rsp+50h] [rbp-290h]
  __int64 v124; // [rsp+50h] [rbp-290h]
  __int64 *i; // [rsp+58h] [rbp-288h]
  __int64 *v127; // [rsp+58h] [rbp-288h]
  __int64 v128; // [rsp+60h] [rbp-280h] BYREF
  __int64 v129; // [rsp+68h] [rbp-278h] BYREF
  __int64 v130; // [rsp+70h] [rbp-270h] BYREF
  __int64 v131; // [rsp+78h] [rbp-268h] BYREF
  __int64 v132; // [rsp+80h] [rbp-260h] BYREF
  __int64 v133; // [rsp+88h] [rbp-258h] BYREF
  __int64 v134; // [rsp+90h] [rbp-250h] BYREF
  __int64 v135; // [rsp+98h] [rbp-248h] BYREF
  _BYTE *v136; // [rsp+A0h] [rbp-240h] BYREF
  __int64 v137; // [rsp+A8h] [rbp-238h]
  _BYTE v138[160]; // [rsp+B0h] [rbp-230h] BYREF
  _BYTE *v139; // [rsp+150h] [rbp-190h] BYREF
  __int64 v140; // [rsp+158h] [rbp-188h]
  _BYTE v141[160]; // [rsp+160h] [rbp-180h] BYREF
  _BYTE *v142; // [rsp+200h] [rbp-E0h] BYREF
  __int64 v143; // [rsp+208h] [rbp-D8h]
  _BYTE v144[208]; // [rsp+210h] [rbp-D0h] BYREF

  v2 = a2;
  v116 = 0;
  v112 = *(_QWORD **)(a2 + 56);
  v114 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v113 = a2 + 24;
  while ( 1 )
  {
    v3 = 1;
    v111 = *(_QWORD *)(v2 + 8);
    if ( *(_DWORD *)(a1 + 96) && *(_QWORD **)(v2 + 8) != v112 + 40 )
    {
      v4 = *(__int64 **)(a1 + 88);
      v5 = *(unsigned int *)(a1 + 104);
      if ( (_DWORD)v5 )
      {
        v6 = v5 - 1;
        v7 = (v5 - 1) & v114;
        v8 = &v4[2 * v7];
        v9 = *v8;
        if ( *v8 != v2 )
        {
          v71 = 1;
          while ( v9 != -8 )
          {
            v81 = v71 + 1;
            v7 = v6 & (v71 + v7);
            v8 = &v4[2 * v7];
            v9 = *v8;
            if ( v2 == *v8 )
              goto LABEL_6;
            v71 = v81;
          }
          v8 = &v4[2 * (unsigned int)v5];
        }
LABEL_6:
        v10 = v6 & (((unsigned int)v111 >> 9) ^ ((unsigned int)v111 >> 4));
        v11 = &v4[2 * v10];
        v12 = *v11;
        if ( v111 == *v11 )
          goto LABEL_7;
        v70 = 1;
        while ( v12 != -8 )
        {
          v82 = v70 + 1;
          v10 = v6 & (v70 + v10);
          v11 = &v4[2 * v10];
          v12 = *v11;
          if ( v111 == *v11 )
            goto LABEL_7;
          v70 = v82;
        }
        v4 += 2 * v5;
      }
      else
      {
        v8 = *(__int64 **)(a1 + 88);
      }
      v11 = v4;
LABEL_7:
      v3 = *((_DWORD *)v8 + 2) == *((_DWORD *)v11 + 2);
    }
    if ( v113 == sub_1DD6100(v2) && !*(_BYTE *)(v2 + 180) && *(_BYTE *)(v2 + 181) != 1 && v3 )
    {
      v35 = *(__int64 **)(v2 + 88);
      v36 = *(_QWORD *)(a1 + 144);
      for ( i = *(__int64 **)(v2 + 96); i != v35; ++v35 )
      {
        v37 = *v35;
        if ( (unsigned int)((__int64)(*(_QWORD *)(*v35 + 72) - *(_QWORD *)(*v35 + 64)) >> 3) == 1 )
        {
          v38 = sub_1DD5D40(*v35, *(_QWORD *)(v37 + 32));
          v39 = *(_QWORD *)(v2 + 32);
          if ( v113 != v39 )
          {
            v119 = v2;
            v40 = v38;
            do
            {
              if ( **(_WORD **)(v39 + 16) == 12 )
                (*(void (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v36 + 160LL))(
                  v36,
                  v37,
                  v40,
                  v39);
              v39 = *(_QWORD *)(v39 + 8);
            }
            while ( v113 != v39 );
            v2 = v119;
          }
        }
      }
      v72 = *(__int64 **)(v2 + 64);
      v73 = v36;
      v127 = *(__int64 **)(v2 + 72);
      if ( v127 != v72 )
      {
        do
        {
          v74 = *v72;
          if ( (unsigned int)((__int64)(*(_QWORD *)(*v72 + 96) - *(_QWORD *)(*v72 + 88)) >> 3) == 1 )
          {
            v75 = sub_1DD5EE0(*v72);
            v76 = *(_QWORD *)(v2 + 32);
            if ( v113 != v76 )
            {
              v121 = v2;
              v77 = v75;
              do
              {
                if ( **(_WORD **)(v76 + 16) == 12 )
                  (*(void (__fastcall **)(__int64, __int64, unsigned __int64, __int64))(*(_QWORD *)v73 + 160LL))(
                    v73,
                    v74,
                    v77,
                    v76);
                v76 = *(_QWORD *)(v76 + 8);
              }
              while ( v113 != v76 );
              v2 = v121;
            }
          }
          ++v72;
        }
        while ( v127 != v72 );
        if ( *(_QWORD *)(v2 + 72) != *(_QWORD *)(v2 + 64) && (_QWORD *)v111 != v112 + 40 && !*(_BYTE *)(v111 + 180) )
        {
          v78 = sub_1DD6970(v2, v111);
          if ( v78 )
          {
            for ( j = *(_QWORD *)(v2 + 72); *(_QWORD *)(v2 + 64) != j; j = *(_QWORD *)(v2 + 72) )
              sub_1DD9680(*(_QWORD *)(j - 8), v2, v111);
            v80 = v112[9];
            if ( v80 )
              sub_1E0A860(v80, v2, v111);
            return v78;
          }
        }
      }
      return v116;
    }
    v13 = *(_QWORD *)v2;
    v128 = 0;
    LOBYTE(v14) = 1;
    v129 = 0;
    v117 = v13 & 0xFFFFFFFFFFFFFFF8LL;
    v136 = v138;
    v137 = 0x400000000LL;
    v15 = *(_QWORD *)(a1 + 144);
    v16 = *(__int64 (**)())(*(_QWORD *)v15 + 264LL);
    if ( v16 == sub_1D820E0 )
      goto LABEL_10;
    LOBYTE(v14) = ((__int64 (__fastcall *)(__int64, unsigned __int64, __int64 *, __int64 *, _BYTE **, __int64))v16)(
                    v15,
                    v117,
                    &v128,
                    &v129,
                    &v136,
                    1);
    if ( (_BYTE)v14 )
      goto LABEL_10;
    v116 |= sub_1DD9390(v117, v128, v129, (_DWORD)v137 != 0);
    if ( !v128 )
    {
      if ( !(_DWORD)v137
        && (unsigned int)((__int64)(*(_QWORD *)(v2 + 72) - *(_QWORD *)(v2 + 64)) >> 3) == 1
        && (unsigned int)((__int64)(*(_QWORD *)(v117 + 96) - *(_QWORD *)(v117 + 88)) >> 3) == 1
        && !*(_BYTE *)(v2 + 181)
        && !*(_BYTE *)(v2 + 180) )
      {
        v55 = v117 + 24;
        if ( v117 + 24 != *(_QWORD *)(v117 + 32) )
        {
          v142 = (_BYTE *)(v117 + 24);
          sub_1F4E0E0((unsigned __int64 *)&v142);
          v56 = *(_QWORD *)(v2 + 32);
          while ( *(_BYTE **)(v117 + 32) != v142
               && v113 != v56
               && (unsigned __int16)(**((_WORD **)v142 + 2) - 12) <= 1u
               && (unsigned __int16)(**(_WORD **)(v56 + 16) - 12) <= 1u
               && (unsigned __int8)sub_1E15D60(v56, (__int64)v142, 0) )
          {
            v57 = v56;
            v58 = v56;
            if ( (*(_BYTE *)v56 & 4) == 0 )
            {
              do
              {
                v101 = *(_WORD *)(v58 + 46);
                v57 = v58;
                v58 = *(_QWORD *)(v58 + 8);
              }
              while ( (v101 & 8) != 0 );
            }
            v59 = *(_QWORD *)(v57 + 8);
            sub_1F4E0E0((unsigned __int64 *)&v142);
            v60 = v56;
            v56 = v59;
            sub_1E16240(v60);
          }
        }
        v102 = *(unsigned __int64 **)(v2 + 32);
        if ( v102 != (unsigned __int64 *)v113 && v55 != v113 )
        {
          if ( v117 + 16 != v2 + 16 )
            sub_1DD5C00((__int64 *)(v117 + 16), v2 + 16, *(_QWORD *)(v2 + 32), v113);
          if ( (unsigned __int64 *)v113 != v102 )
          {
            v103 = *(_QWORD *)(v2 + 24) & 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)((*v102 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v113;
            *(_QWORD *)(v2 + 24) = *(_QWORD *)(v2 + 24) & 7LL | *v102 & 0xFFFFFFFFFFFFFFF8LL;
            v104 = *(_QWORD *)(v117 + 24);
            *(_QWORD *)(v103 + 8) = v55;
            *v102 = v104 & 0xFFFFFFFFFFFFFFF8LL | *v102 & 7;
            *(_QWORD *)((v104 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v102;
            *(_QWORD *)(v117 + 24) = v103 | *(_QWORD *)(v117 + 24) & 7LL;
          }
        }
        sub_1DD9130(v117, *(__int64 **)(v117 + 88), 0);
        sub_1DD91F0(v117, (_QWORD *)v2);
        v116 = 1;
        goto LABEL_55;
      }
      v43 = v129;
LABEL_81:
      if ( v2 != v43 )
        goto LABEL_82;
      sub_20D65A0((__int64 *)&v142, v117);
      (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD))(**(_QWORD **)(a1 + 144) + 280LL))(
        *(_QWORD *)(a1 + 144),
        v117,
        0);
      (*(void (__fastcall **)(_QWORD, unsigned __int64, __int64, _QWORD, _BYTE *, _QWORD, _BYTE **, _QWORD))(**(_QWORD **)(a1 + 144) + 288LL))(
        *(_QWORD *)(a1 + 144),
        v117,
        v128,
        0,
        v136,
        (unsigned int)v137,
        &v142,
        0);
      if ( !v142 )
        goto LABEL_43;
LABEL_139:
      sub_161E7C0((__int64)&v142, (__int64)v142);
      goto LABEL_43;
    }
    v43 = v129;
    if ( v128 == v129 )
    {
      sub_20D65A0((__int64 *)&v142, v117);
      (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD))(**(_QWORD **)(a1 + 144) + 280LL))(
        *(_QWORD *)(a1 + 144),
        v117,
        0);
      LODWORD(v137) = 0;
      if ( v128 != v2 )
        (*(void (__fastcall **)(_QWORD, unsigned __int64, __int64, _QWORD, _BYTE *, _QWORD, _BYTE **, _QWORD))(**(_QWORD **)(a1 + 144) + 288LL))(
          *(_QWORD *)(a1 + 144),
          v117,
          v128,
          0,
          v136,
          0,
          &v142,
          0);
      if ( !v142 )
        goto LABEL_43;
      goto LABEL_139;
    }
    if ( v128 != v2 )
      goto LABEL_81;
    if ( !v129 )
    {
      (*(void (__fastcall **)(_QWORD, unsigned __int64))(**(_QWORD **)(a1 + 144) + 280LL))(*(_QWORD *)(a1 + 144), v117);
      goto LABEL_43;
    }
    v142 = v144;
    v143 = 0x400000000LL;
    if ( (_DWORD)v137 )
    {
      sub_20D61B0((__int64)&v142, (__int64)&v136, v129, v41, v42, v137);
      v61 = *(_QWORD *)(a1 + 144);
      v62 = *(__int64 (**)())(*(_QWORD *)v61 + 624LL);
      if ( v62 != sub_1D918B0 )
      {
LABEL_123:
        if ( !((unsigned __int8 (__fastcall *)(__int64, _BYTE **))v62)(v61, &v142) )
        {
          sub_20D65A0((__int64 *)&v139, v117);
          (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD))(**(_QWORD **)(a1 + 144) + 280LL))(
            *(_QWORD *)(a1 + 144),
            v117,
            0);
          (*(void (__fastcall **)(_QWORD, unsigned __int64, __int64, _QWORD, _BYTE *, _QWORD, _BYTE **, _QWORD))(**(_QWORD **)(a1 + 144) + 288LL))(
            *(_QWORD *)(a1 + 144),
            v117,
            v129,
            0,
            v142,
            (unsigned int)v143,
            &v139,
            0);
          if ( v139 )
            sub_161E7C0((__int64)&v139, (__int64)v139);
          v30 = (unsigned __int64)v142;
          if ( v142 != v144 )
            goto LABEL_42;
          goto LABEL_43;
        }
      }
      if ( v142 != v144 )
        _libc_free((unsigned __int64)v142);
      goto LABEL_82;
    }
    v61 = *(_QWORD *)(a1 + 144);
    v62 = *(__int64 (**)())(*(_QWORD *)v61 + 624LL);
    if ( v62 != sub_1D918B0 )
      goto LABEL_123;
LABEL_82:
    if ( *(_QWORD *)(v2 + 96) == *(_QWORD *)(v2 + 88)
      && (_DWORD)v137
      && !v129
      && v128 == v111
      && !sub_1DD6C00((__int64 *)v2) )
    {
      if ( v111 != (v112[40] & 0xFFFFFFFFFFFFFFF8LL) )
        goto LABEL_88;
      v122 = v128;
      v87 = sub_1DD6160(v128);
      v88 = sub_1DD6160(v2);
      v89 = v88;
      if ( v87 == v122 + 24 || v113 == v88 )
        goto LABEL_90;
      LOBYTE(v14) = sub_1DD6970(v122, v2);
      if ( (_BYTE)v14 )
        goto LABEL_88;
      if ( !sub_1DD6970(v2, v122) )
      {
        v90 = *(_WORD *)(v89 + 46);
        if ( (v90 & 4) != 0 || (v90 & 8) == 0 )
          v14 = (*(_QWORD *)(*(_QWORD *)(v89 + 16) + 8LL) >> 4) & 1LL;
        else
          LOBYTE(v14) = sub_1E15D00(v89, 0x10u, 1);
        if ( (_BYTE)v14 )
        {
          v91 = *(_WORD *)(v87 + 46);
          if ( (v91 & 4) != 0 || (v91 & 8) == 0 )
            v92 = (*(_QWORD *)(*(_QWORD *)(v87 + 16) + 8LL) >> 4) & 1LL;
          else
            LOBYTE(v92) = sub_1E15D00(v87, 0x10u, 1);
          if ( (_BYTE)v92 )
            goto LABEL_90;
LABEL_88:
          v142 = v144;
          v143 = 0x400000000LL;
          if ( (_DWORD)v137 )
          {
            sub_20D61B0((__int64)&v142, (__int64)&v136, v44, (unsigned int)v137, v45, v46);
            v47 = *(_QWORD *)(a1 + 144);
            v48 = *(__int64 (**)())(*(_QWORD *)v47 + 624LL);
            if ( v48 != sub_1D918B0 )
              goto LABEL_248;
          }
          else
          {
            v47 = *(_QWORD *)(a1 + 144);
            v48 = *(__int64 (**)())(*(_QWORD *)v47 + 624LL);
            if ( v48 == sub_1D918B0 )
            {
LABEL_90:
              LOBYTE(v14) = 0;
              goto LABEL_10;
            }
LABEL_248:
            if ( !((unsigned __int8 (__fastcall *)(__int64, _BYTE **))v48)(v47, &v142) )
            {
              sub_20D65A0((__int64 *)&v139, v117);
              (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD))(**(_QWORD **)(a1 + 144) + 280LL))(
                *(_QWORD *)(a1 + 144),
                v117,
                0);
              (*(void (__fastcall **)(_QWORD, unsigned __int64, __int64, _QWORD, _BYTE *, _QWORD, _BYTE **, _QWORD))(**(_QWORD **)(a1 + 144) + 288LL))(
                *(_QWORD *)(a1 + 144),
                v117,
                v2,
                0,
                v142,
                (unsigned int)v143,
                &v139,
                0);
              sub_1DD6900((__int64 *)v2, v112[40] & 0xFFFFFFFFFFFFFFF8LL);
              sub_17CD270((__int64 *)&v139);
              if ( v142 != v144 )
                _libc_free((unsigned __int64)v142);
              v116 = 1;
              goto LABEL_55;
            }
          }
          if ( v142 != v144 )
            _libc_free((unsigned __int64)v142);
          goto LABEL_90;
        }
      }
    }
LABEL_10:
    if ( v113 != sub_1DD6100(v2)
      && (unsigned int)((__int64)(*(_QWORD *)(v2 + 72) - *(_QWORD *)(v2 + 64)) >> 3) == 1
      && ((v63 = *v112 + 112LL, (unsigned __int8)sub_1560180(v63, 34)) || (unsigned __int8)sub_1560180(v63, 17)) )
    {
      v64 = sub_1DD6100(v2);
      v17 = *(_QWORD *)(a1 + 144);
      v65 = *(__int64 (**)())(*(_QWORD *)v17 + 672LL);
      if ( v65 != sub_1F394D0 )
      {
        if ( !((unsigned __int8 (__fastcall *)(__int64, __int64))v65)(v17, v64) )
          goto LABEL_137;
        v66 = **(_QWORD **)(v2 + 64);
        v135 = 0;
        v143 = 0x400000000LL;
        v139 = 0;
        v17 = *(_QWORD *)(a1 + 144);
        v142 = v144;
        v67 = *(__int64 (**)())(*(_QWORD *)v17 + 264LL);
        if ( v67 != sub_1D820E0 )
        {
          v120 = v66;
          if ( !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *, _BYTE **, _BYTE **, __int64))v67)(
                  v17,
                  v66,
                  &v135,
                  &v139,
                  &v142,
                  1) )
          {
            if ( (_DWORD)v143 )
            {
              if ( v135 == v2 && v139 != (_BYTE *)v2 )
              {
                v83 = *(_QWORD *)(a1 + 144);
                v84 = *(__int64 (**)())(*(_QWORD *)v83 + 680LL);
                if ( v84 != sub_1F394E0 )
                {
                  v85 = ((__int64 (__fastcall *)(__int64, _BYTE **, __int64))v84)(v83, &v142, v64);
                  if ( v85 )
                  {
                    v86 = v85;
                    (*(void (__fastcall **)(_QWORD, __int64, _BYTE **, __int64))(**(_QWORD **)(a1 + 144) + 688LL))(
                      *(_QWORD *)(a1 + 144),
                      v120,
                      &v142,
                      v64);
                    sub_1DD91B0(v120, v2);
                    if ( v142 != v144 )
                      _libc_free((unsigned __int64)v142);
                    v116 = v86;
                    goto LABEL_55;
                  }
                }
              }
            }
          }
          if ( v142 != v144 )
            _libc_free((unsigned __int64)v142);
LABEL_137:
          v17 = *(_QWORD *)(a1 + 144);
        }
      }
    }
    else
    {
      v17 = *(_QWORD *)(a1 + 144);
    }
    v115 = 1;
    v139 = v141;
    v140 = 0x400000000LL;
    v130 = 0;
    v131 = 0;
    v18 = *(__int64 (**)())(*(_QWORD *)v17 + 264LL);
    if ( v18 == sub_1D820E0 )
      goto LABEL_14;
    v115 = ((__int64 (__fastcall *)(__int64, __int64, __int64 *, __int64 *, _BYTE **, __int64))v18)(
             v17,
             v2,
             &v130,
             &v131,
             &v139,
             1);
    if ( v115 )
      goto LABEL_14;
    v116 |= sub_1DD9390(v2, v130, v131, (_DWORD)v140 != 0);
    if ( !v130 )
      goto LABEL_14;
    if ( v130 != v2 && v131 == v2 && v131 )
    {
      v143 = 0x400000000LL;
      v142 = v144;
      if ( (_DWORD)v140 )
      {
        LOBYTE(v49) = v131 == v2;
        sub_20D61B0((__int64)&v142, (__int64)&v139, v131, v49, v50, v51);
        v68 = *(_QWORD *)(a1 + 144);
        v69 = *(__int64 (**)())(*(_QWORD *)v68 + 624LL);
        if ( v69 != sub_1D918B0 )
        {
LABEL_144:
          if ( !((unsigned __int8 (__fastcall *)(__int64, _BYTE **))v69)(v68, &v142) )
          {
            sub_20D65A0(&v135, v2);
            (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 144) + 280LL))(
              *(_QWORD *)(a1 + 144),
              v2,
              0);
            (*(void (__fastcall **)(_QWORD, __int64, __int64, __int64, _BYTE *, _QWORD, __int64 *, _QWORD))(**(_QWORD **)(a1 + 144) + 288LL))(
              *(_QWORD *)(a1 + 144),
              v2,
              v131,
              v130,
              v142,
              (unsigned int)v143,
              &v135,
              0);
            if ( v135 )
              sub_161E7C0((__int64)&v135, v135);
            goto LABEL_60;
          }
        }
        if ( v142 != v144 )
          _libc_free((unsigned __int64)v142);
        if ( !v130 )
          goto LABEL_14;
        goto LABEL_95;
      }
      v68 = *(_QWORD *)(a1 + 144);
      v69 = *(__int64 (**)())(*(_QWORD *)v68 + 624LL);
      if ( v69 != sub_1D918B0 )
        goto LABEL_144;
    }
LABEL_95:
    if ( !(_DWORD)v140 && !v131 )
    {
      v52 = sub_1DD6100(v2);
      v53 = *(_WORD *)(v52 + 46);
      if ( (v53 & 4) != 0 || (v53 & 8) == 0 )
        v54 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v52 + 16) + 8LL) >> 7;
      else
        v54 = sub_1E15D00(v52, 0x80u, 1);
      if ( !v54 )
      {
LABEL_104:
        v115 = 0;
        goto LABEL_14;
      }
      if ( v130 != v2 && !*(_BYTE *)(v2 + 181) )
      {
        v115 = *(_BYTE *)(v2 + 180);
        if ( v115 )
          goto LABEL_104;
        sub_20D65A0(&v132, v2);
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 144) + 280LL))(
          *(_QWORD *)(a1 + 144),
          v2,
          0);
        if ( v113 == sub_1DD6100(v2) )
        {
          v105 = *(_QWORD **)(v2 + 32);
          if ( v105 != (_QWORD *)v113 )
          {
            v124 = v2;
            v106 = v2 + 16;
            do
            {
              v107 = v105;
              v105 = (_QWORD *)v105[1];
              sub_1DD5BC0(v106, (__int64)v107);
              v108 = (unsigned __int64 *)v107[1];
              v109 = *v107 & 0xFFFFFFFFFFFFFFF8LL;
              *v108 = v109 | *v108 & 7;
              *(_QWORD *)(v109 + 8) = v108;
              *v107 &= 7uLL;
              v107[1] = 0;
              sub_1DD5C20(v106);
            }
            while ( (_QWORD *)v113 != v105 );
            v2 = v124;
          }
        }
        if ( v113 != (*(_QWORD *)(v2 + 24) & 0xFFFFFFFFFFFFFFF8LL) )
          goto LABEL_201;
        if ( sub_1DD6C00((__int64 *)v117) )
        {
          if ( !(_BYTE)v14 || !sub_1DD6970(v117, v2) )
          {
            if ( sub_1DD6970(v117, v2) && v128 != v2 && v129 != v2 )
            {
              if ( v128 )
                v129 = v2;
              else
                v128 = v2;
              sub_20D65A0((__int64 *)&v142, v117);
              (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD))(**(_QWORD **)(a1 + 144) + 280LL))(
                *(_QWORD *)(a1 + 144),
                v117,
                0);
              (*(void (__fastcall **)(_QWORD, unsigned __int64, __int64, __int64, _BYTE *, _QWORD, _BYTE **, _QWORD))(**(_QWORD **)(a1 + 144) + 288LL))(
                *(_QWORD *)(a1 + 144),
                v117,
                v128,
                v129,
                v136,
                (unsigned int)v137,
                &v142,
                0);
              sub_17CD270((__int64 *)&v142);
            }
            goto LABEL_228;
          }
LABEL_201:
          (*(void (__fastcall **)(_QWORD, __int64, __int64, _QWORD, _BYTE *, _QWORD, __int64 *, _QWORD))(**(_QWORD **)(a1 + 144) + 288LL))(
            *(_QWORD *)(a1 + 144),
            v2,
            v130,
            0,
            v139,
            (unsigned int)v140,
            &v132,
            0);
          if ( v132 )
            sub_161E7C0((__int64)&v132, v132);
          goto LABEL_14;
        }
LABEL_228:
        v93 = *(_QWORD *)(v2 + 72);
        v94 = *(_QWORD *)(v2 + 64);
        if ( (unsigned int)((v93 - v94) >> 3) )
        {
          v95 = v116;
          v116 = 0;
          v96 = 0;
          v110 = v54;
          v123 = 0;
          do
          {
            v98 = *(_QWORD *)(v94 + 8 * v123);
            if ( v2 == v98 )
            {
              ++v123;
              v116 = v110;
            }
            else
            {
              sub_1DD9680(v98, v2, v130);
              v97 = *(_QWORD *)(a1 + 144);
              v142 = v144;
              v143 = 0x400000000LL;
              v133 = 0;
              v134 = 0;
              if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64 *, __int64 *, _BYTE **, __int64))(*(_QWORD *)v97 + 264LL))(
                      v97,
                      v98,
                      &v133,
                      &v134,
                      &v142,
                      1)
                && v133
                && v133 == v134 )
              {
                sub_20D65A0(&v135, v98);
                (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 144) + 280LL))(
                  *(_QWORD *)(a1 + 144),
                  v98,
                  0);
                v100 = *(_QWORD *)(a1 + 144);
                LODWORD(v143) = 0;
                (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD, _BYTE *, _QWORD, __int64 *, _QWORD))(*(_QWORD *)v100 + 288LL))(
                  v100,
                  v98,
                  v133,
                  0,
                  v142,
                  0,
                  &v135,
                  0);
                sub_1DD9390(v98, v133, 0, 0);
                sub_17CD270(&v135);
                v95 = v110;
              }
              if ( v142 != v144 )
                _libc_free((unsigned __int64)v142);
              v96 = v110;
              v93 = *(_QWORD *)(v2 + 72);
              v94 = *(_QWORD *)(v2 + 64);
            }
          }
          while ( (unsigned int)((v93 - v94) >> 3) != v123 );
          v99 = v112[9];
          if ( v99 )
            goto LABEL_241;
        }
        else
        {
          v99 = v112[9];
          if ( !v99 )
            goto LABEL_201;
          v95 = v116;
          v96 = 0;
          v116 = 0;
LABEL_241:
          sub_1E0A860(v99, v2, v130);
        }
        if ( v96 )
        {
          if ( !v116 )
          {
            sub_17CD270(&v132);
            v116 = v96;
            goto LABEL_53;
          }
        }
        else
        {
          v116 = v95;
        }
        goto LABEL_201;
      }
    }
LABEL_14:
    if ( sub_1DD6C00((__int64 *)v117) )
      goto LABEL_53;
    v118 = sub_1DD6C00((__int64 *)v2);
    if ( *(_BYTE *)(v2 + 180) )
      break;
    v19 = *(__int64 **)(v2 + 72);
    if ( v19 == *(__int64 **)(v2 + 64) )
      break;
    v20 = *(__int64 **)(v2 + 64);
    while ( 1 )
    {
      v21 = *v20;
      v133 = 0;
      v134 = 0;
      v142 = v144;
      v143 = 0x400000000LL;
      if ( v2 != v21 )
        break;
LABEL_20:
      if ( v19 == ++v20 )
        goto LABEL_31;
    }
    if ( sub_1DD6C00((__int64 *)v21) )
      goto LABEL_18;
    v22 = *(_QWORD *)(a1 + 144);
    v23 = *(__int64 (**)())(*(_QWORD *)v22 + 264LL);
    if ( v23 == sub_1D820E0
      || ((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *, __int64 *, _BYTE **, __int64))v23)(
           v22,
           v21,
           &v133,
           &v134,
           &v142,
           1) )
    {
      goto LABEL_18;
    }
    if ( !v118 )
    {
      v24 = v21;
      goto LABEL_59;
    }
    if ( v130 && v131 || *(_DWORD *)(v21 + 48) > *(_DWORD *)(v2 + 48) )
    {
LABEL_18:
      if ( v142 != v144 )
        _libc_free((unsigned __int64)v142);
      goto LABEL_20;
    }
    v24 = v21;
    v25 = *(_QWORD *)(v2 + 8);
    LODWORD(v140) = 0;
    sub_20D65A0(&v135, v2);
    (*(void (__fastcall **)(_QWORD, __int64, __int64, _QWORD, _BYTE *, _QWORD, __int64 *, _QWORD))(**(_QWORD **)(a1 + 144)
                                                                                                 + 288LL))(
      *(_QWORD *)(a1 + 144),
      v2,
      v25,
      0,
      v139,
      (unsigned int)v140,
      &v135,
      0);
    if ( v135 )
      sub_161E7C0((__int64)&v135, v135);
LABEL_59:
    sub_1DD6900((__int64 *)v2, v24);
LABEL_60:
    if ( v142 != v144 )
      _libc_free((unsigned __int64)v142);
LABEL_41:
    v30 = (unsigned __int64)v139;
    if ( v139 != v141 )
LABEL_42:
      _libc_free(v30);
LABEL_43:
    if ( v136 != v138 )
      _libc_free((unsigned __int64)v136);
    v116 = 1;
  }
LABEL_31:
  if ( !v118 )
  {
    v26 = *(_QWORD ***)(v2 + 96);
    v27 = *(_QWORD ***)(v2 + 88);
    if ( v26 != v27 )
    {
      while ( 1 )
      {
        v28 = *v27;
        v29 = (__int64 *)(**v27 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (__int64 *)v2 != v29 && (_QWORD *)v2 != v28 && !sub_1DD6C00(v29) && !v115 && !*((_BYTE *)v28 + 180) )
          break;
        if ( v26 == ++v27 )
          goto LABEL_46;
      }
      sub_1DD6890((__int64 *)v2, v28);
      goto LABEL_41;
    }
LABEL_46:
    v134 = 0;
    v143 = 0x400000000LL;
    v135 = 0;
    v142 = v144;
    if ( (_QWORD *)v111 != v112 + 40 && !*(_BYTE *)(v111 + 180) )
    {
      v31 = *(_QWORD *)(a1 + 144);
      v32 = *(__int64 (**)())(*(_QWORD *)v31 + 264LL);
      if ( v32 != sub_1D820E0 )
      {
        if ( !((unsigned __int8 (__fastcall *)(__int64, unsigned __int64, __int64 *, __int64 *, _BYTE **, __int64))v32)(
                v31,
                v117,
                &v134,
                &v135,
                &v142,
                1)
          && (v33 = sub_1DD6970(v117, v111)) )
        {
          sub_1DD6900((__int64 *)v2, v112[40] & 0xFFFFFFFFFFFFFFF8LL);
          if ( v142 != v144 )
            _libc_free((unsigned __int64)v142);
          v116 = v33;
        }
        else if ( v142 != v144 )
        {
          _libc_free((unsigned __int64)v142);
        }
      }
    }
  }
LABEL_53:
  if ( v139 != v141 )
    _libc_free((unsigned __int64)v139);
LABEL_55:
  if ( v136 != v138 )
    _libc_free((unsigned __int64)v136);
  return v116;
}
