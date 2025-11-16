// Function: sub_24955E0
// Address: 0x24955e0
//
void __fastcall sub_24955E0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v5; // r12
  __int64 v6; // rbx
  char *v7; // r15
  __int64 v8; // rax
  __int64 v9; // rbx
  _QWORD *v10; // r14
  char *v11; // rbx
  __int64 v12; // r12
  int v13; // eax
  int v14; // eax
  __int64 *v15; // rdi
  __int64 *v16; // rdi
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rbx
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 v22; // r14
  _QWORD *v23; // rax
  __int64 v24; // r9
  __int64 v25; // r12
  __int64 v26; // r14
  _BYTE *v27; // rbx
  __int64 v28; // rdx
  unsigned int v29; // esi
  __int64 v30; // rdx
  unsigned __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // r12
  int v34; // eax
  __int64 *v35; // rdi
  __int64 v36; // r12
  int v37; // eax
  __int64 *v38; // rdi
  __int64 v39; // r12
  int v40; // eax
  __int64 *v41; // rdi
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rsi
  __int64 v48; // r9
  __int64 v49; // r8
  __int64 v50; // rax
  int v51; // ecx
  _QWORD *v52; // rdx
  __int64 v53; // r15
  __int64 v54; // rax
  unsigned __int64 *v55; // r10
  __int64 v56; // rbx
  __int64 v57; // r15
  unsigned __int64 *v58; // r12
  __int64 *v59; // rdi
  __int64 v60; // rax
  __int64 v61; // r8
  __int64 v62; // r9
  unsigned __int8 *v63; // rsi
  unsigned __int8 *v64; // r11
  __int64 v65; // rax
  unsigned __int64 v66; // rdx
  int v67; // eax
  __int64 v68; // rax
  __int64 v69; // rcx
  __int64 v70; // rdi
  unsigned int v71; // edx
  unsigned __int8 **v72; // rax
  unsigned __int8 *v73; // r11
  const char *v74; // rax
  unsigned __int64 v75; // rdx
  int v76; // eax
  const char *v77; // rax
  __int64 v78; // rdx
  __int64 v79; // rdx
  int v80; // eax
  unsigned __int64 v81; // rsi
  __int64 v82; // [rsp+28h] [rbp-1E8h]
  __int64 v83; // [rsp+28h] [rbp-1E8h]
  _QWORD *v85; // [rsp+30h] [rbp-1E0h]
  __int64 v87; // [rsp+38h] [rbp-1D8h]
  __int64 v88; // [rsp+38h] [rbp-1D8h]
  unsigned __int8 *v89; // [rsp+38h] [rbp-1D8h]
  unsigned int v90; // [rsp+48h] [rbp-1C8h]
  char v91; // [rsp+48h] [rbp-1C8h]
  __int64 v92; // [rsp+50h] [rbp-1C0h]
  char *v93; // [rsp+58h] [rbp-1B8h]
  __int64 v94; // [rsp+58h] [rbp-1B8h]
  unsigned __int64 *v95; // [rsp+58h] [rbp-1B8h]
  __int64 v96; // [rsp+58h] [rbp-1B8h]
  __int64 v97; // [rsp+60h] [rbp-1B0h]
  __int64 v98; // [rsp+68h] [rbp-1A8h]
  __int64 v99; // [rsp+70h] [rbp-1A0h]
  __int64 v100; // [rsp+78h] [rbp-198h]
  __int64 v101; // [rsp+80h] [rbp-190h]
  __int64 v102; // [rsp+88h] [rbp-188h]
  __int64 v103; // [rsp+90h] [rbp-180h] BYREF
  char v104; // [rsp+98h] [rbp-178h]
  char v105[32]; // [rsp+A0h] [rbp-170h] BYREF
  __int16 v106; // [rsp+C0h] [rbp-150h]
  _QWORD v107[4]; // [rsp+D0h] [rbp-140h] BYREF
  __int16 v108; // [rsp+F0h] [rbp-120h]
  _BYTE *v109; // [rsp+100h] [rbp-110h] BYREF
  __int64 v110; // [rsp+108h] [rbp-108h]
  _BYTE v111[64]; // [rsp+110h] [rbp-100h] BYREF
  _BYTE *v112; // [rsp+150h] [rbp-C0h] BYREF
  __int64 v113; // [rsp+158h] [rbp-B8h]
  _BYTE v114[32]; // [rsp+160h] [rbp-B0h] BYREF
  __int64 v115; // [rsp+180h] [rbp-90h]
  __int64 v116; // [rsp+188h] [rbp-88h]
  __int64 v117; // [rsp+190h] [rbp-80h]
  __int64 v118; // [rsp+198h] [rbp-78h]
  void **v119; // [rsp+1A0h] [rbp-70h]
  void **v120; // [rsp+1A8h] [rbp-68h]
  __int64 v121; // [rsp+1B0h] [rbp-60h]
  int v122; // [rsp+1B8h] [rbp-58h]
  __int16 v123; // [rsp+1BCh] [rbp-54h]
  char v124; // [rsp+1BEh] [rbp-52h]
  __int64 v125; // [rsp+1C0h] [rbp-50h]
  __int64 v126; // [rsp+1C8h] [rbp-48h]
  void *v127; // [rsp+1D0h] [rbp-40h] BYREF
  void *v128; // [rsp+1D8h] [rbp-38h] BYREF

  if ( **(_BYTE **)(a2 - 32) == 25 )
    return;
  v92 = a2;
  v5 = a2;
  v6 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v7 = *(char **)(a2 - 8);
    v93 = &v7[v6];
  }
  else
  {
    v93 = (char *)a2;
    v7 = (char *)(a2 - v6);
  }
  v8 = v6 >> 5;
  v9 = v6 >> 7;
  if ( v9 )
  {
    v10 = (_QWORD *)(a1 + 16);
    v11 = &v7[128 * v9];
    while ( 1 )
    {
      v12 = *(_QWORD *)(*(_QWORD *)v7 + 8LL);
      v13 = *(unsigned __int8 *)(v12 + 8);
      switch ( (_BYTE)v13 )
      {
        case 2:
          v32 = 1;
          goto LABEL_42;
        case 3:
          v32 = 2;
          goto LABEL_42;
        case 4:
          v32 = 3;
LABEL_42:
          if ( (**(__int64 (__fastcall ***)(_QWORD, _QWORD))v10[v32])(v10[v32], *(_QWORD *)(a1 + 16)) )
            goto LABEL_93;
          goto LABEL_43;
      }
      if ( (unsigned int)(v13 - 17) > 1 || sub_BCEA30(*(_QWORD *)(*(_QWORD *)v7 + 8LL)) )
        goto LABEL_43;
      v14 = *(unsigned __int8 *)(*(_QWORD *)(v12 + 24) + 8LL);
      switch ( (_BYTE)v14 )
      {
        case 2:
          v45 = 1;
          break;
        case 3:
          v45 = 2;
          break;
        case 4:
          v45 = 3;
          break;
        default:
          if ( (unsigned int)(v14 - 17) > 1 )
            goto LABEL_43;
          v82 = *(_QWORD *)(v12 + 24);
          if ( sub_BCEA30(v82) )
            goto LABEL_43;
          v15 = (__int64 *)sub_2491640((_QWORD *)(a1 + 16), *(_QWORD *)(v82 + 24));
          if ( !v15 )
            goto LABEL_43;
          BYTE4(v98) = *(_BYTE *)(v82 + 8) == 18;
          LODWORD(v98) = *(_DWORD *)(v82 + 32);
          v16 = (__int64 *)sub_BCE1B0(v15, v98);
          goto LABEL_91;
      }
      v16 = (__int64 *)(**(__int64 (__fastcall ***)(_QWORD, _QWORD))v10[v45])(v10[v45], *(_QWORD *)(a1 + 16));
LABEL_91:
      if ( v16 )
      {
        BYTE4(v97) = *(_BYTE *)(v12 + 8) == 18;
        LODWORD(v97) = *(_DWORD *)(v12 + 32);
        if ( sub_BCE1B0(v16, v97) )
        {
LABEL_93:
          v5 = a2;
          goto LABEL_94;
        }
      }
LABEL_43:
      v33 = *(_QWORD *)(*((_QWORD *)v7 + 4) + 8LL);
      v34 = *(unsigned __int8 *)(v33 + 8);
      switch ( (_BYTE)v34 )
      {
        case 2:
          v42 = 1;
          goto LABEL_73;
        case 3:
          v42 = 2;
          goto LABEL_73;
        case 4:
          v42 = 3;
LABEL_73:
          if ( (**(__int64 (__fastcall ***)(_QWORD, _QWORD))v10[v42])(v10[v42], *(_QWORD *)(a1 + 16)) )
            goto LABEL_74;
          goto LABEL_50;
      }
      if ( (unsigned int)(v34 - 17) <= 1 && !sub_BCEA30(*(_QWORD *)(*((_QWORD *)v7 + 4) + 8LL)) )
      {
        v35 = (__int64 *)sub_2491640((_QWORD *)(a1 + 16), *(_QWORD *)(v33 + 24));
        if ( v35 )
        {
          BYTE4(v99) = *(_BYTE *)(v33 + 8) == 18;
          LODWORD(v99) = *(_DWORD *)(v33 + 32);
          if ( sub_BCE1B0(v35, v99) )
          {
LABEL_74:
            v5 = a2;
            v7 += 32;
            goto LABEL_94;
          }
        }
      }
LABEL_50:
      v36 = *(_QWORD *)(*((_QWORD *)v7 + 8) + 8LL);
      v37 = *(unsigned __int8 *)(v36 + 8);
      switch ( (_BYTE)v37 )
      {
        case 2:
          v43 = 1;
          goto LABEL_76;
        case 3:
          v43 = 2;
          goto LABEL_76;
        case 4:
          v43 = 3;
LABEL_76:
          if ( (**(__int64 (__fastcall ***)(_QWORD, _QWORD))v10[v43])(v10[v43], *(_QWORD *)(a1 + 16)) )
            goto LABEL_77;
          goto LABEL_57;
      }
      if ( (unsigned int)(v37 - 17) <= 1 && !sub_BCEA30(*(_QWORD *)(*((_QWORD *)v7 + 8) + 8LL)) )
      {
        v38 = (__int64 *)sub_2491640((_QWORD *)(a1 + 16), *(_QWORD *)(v36 + 24));
        if ( v38 )
        {
          BYTE4(v100) = *(_BYTE *)(v36 + 8) == 18;
          LODWORD(v100) = *(_DWORD *)(v36 + 32);
          if ( sub_BCE1B0(v38, v100) )
          {
LABEL_77:
            v5 = a2;
            v7 += 64;
            goto LABEL_94;
          }
        }
      }
LABEL_57:
      v39 = *(_QWORD *)(*((_QWORD *)v7 + 12) + 8LL);
      v40 = *(unsigned __int8 *)(v39 + 8);
      switch ( (_BYTE)v40 )
      {
        case 2:
          v44 = 1;
          break;
        case 3:
          v44 = 2;
          break;
        case 4:
          v44 = 3;
          break;
        default:
          if ( (unsigned int)(v40 - 17) <= 1 && !sub_BCEA30(*(_QWORD *)(*((_QWORD *)v7 + 12) + 8LL)) )
          {
            v41 = (__int64 *)sub_2491640((_QWORD *)(a1 + 16), *(_QWORD *)(v39 + 24));
            if ( v41 )
            {
              BYTE4(v101) = *(_BYTE *)(v39 + 8) == 18;
              LODWORD(v101) = *(_DWORD *)(v39 + 32);
              if ( sub_BCE1B0(v41, v101) )
              {
LABEL_80:
                v5 = a2;
                v7 += 96;
                goto LABEL_94;
              }
            }
          }
          goto LABEL_64;
      }
      if ( (**(__int64 (__fastcall ***)(_QWORD, _QWORD))v10[v44])(v10[v44], *(_QWORD *)(a1 + 16)) )
        goto LABEL_80;
LABEL_64:
      v7 += 128;
      if ( v11 == v7 )
      {
        v5 = a2;
        v8 = (v93 - v7) >> 5;
        break;
      }
    }
  }
  switch ( v8 )
  {
    case 2LL:
      v10 = (_QWORD *)(a1 + 16);
      break;
    case 3LL:
      v10 = (_QWORD *)(a1 + 16);
      if ( sub_2491640((_QWORD *)(a1 + 16), *(_QWORD *)(*(_QWORD *)v7 + 8LL)) )
      {
LABEL_94:
        if ( v7 == v93 )
          return;
        v124 = 7;
        v118 = sub_BD5C60(v5);
        v119 = &v127;
        v120 = &v128;
        v112 = v114;
        v127 = &unk_49DA100;
        v113 = 0x200000000LL;
        v121 = 0;
        v128 = &unk_49DA0B0;
        v46 = *(_QWORD *)(v5 + 40);
        v122 = 0;
        v115 = v46;
        v123 = 512;
        v125 = 0;
        v126 = 0;
        v116 = v5 + 24;
        LOWORD(v117) = 0;
        v47 = *(_QWORD *)sub_B46C60(v5);
        v109 = (_BYTE *)v47;
        if ( v47 && (sub_B96E90((__int64)&v109, v47, 1), (v49 = (__int64)v109) != 0) )
        {
          v50 = (__int64)v112;
          v51 = v113;
          v52 = &v112[16 * (unsigned int)v113];
          if ( v112 != (_BYTE *)v52 )
          {
            while ( *(_DWORD *)v50 )
            {
              v50 += 16;
              if ( v52 == (_QWORD *)v50 )
                goto LABEL_141;
            }
            *(_QWORD *)(v50 + 8) = v109;
LABEL_102:
            sub_B91220((__int64)&v109, v49);
LABEL_103:
            v53 = *(_QWORD *)(v5 - 32);
            v109 = v111;
            v110 = 0x800000000LL;
            v91 = *(_BYTE *)(a1 + 488);
            if ( !v53 || *(_BYTE *)v53 || *(_QWORD *)(v53 + 24) != *(_QWORD *)(v5 + 80) )
            {
              v91 ^= 1u;
              goto LABEL_107;
            }
            if ( v91 )
            {
              v77 = sub_BD5D20(v53);
              v91 = sub_C89090((_QWORD *)(a1 + 472), v77, v78, 0, 0);
            }
            else
            {
              v74 = sub_BD5D20(v53);
              if ( v75 > 6 && *(_DWORD *)v74 == 1936613215 && *((_WORD *)v74 + 2) == 28257 && v74[6] == 95 )
                goto LABEL_107;
              v76 = *(_DWORD *)(v53 + 36);
              LODWORD(v107[0]) = 523;
              if ( v76 )
              {
                if ( v76 != 170 )
                {
LABEL_151:
                  v91 = 1;
                  goto LABEL_107;
                }
              }
              else if ( !sub_981210(*a3, v53, (unsigned int *)v107) || (unsigned int)(LODWORD(v107[0]) - 239) > 2 )
              {
                goto LABEL_151;
              }
              v79 = *(_QWORD *)(v5 + 16);
              if ( !v79 )
                goto LABEL_151;
              while ( (unsigned __int8)(**(_BYTE **)(v79 + 24) - 82) > 1u )
              {
                v79 = *(_QWORD *)(v79 + 8);
                if ( !v79 )
                  goto LABEL_151;
              }
            }
LABEL_107:
            v54 = 4LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF);
            if ( (*(_BYTE *)(v5 + 7) & 0x40) != 0 )
            {
              v55 = *(unsigned __int64 **)(v5 - 8);
              v95 = &v55[v54];
            }
            else
            {
              v95 = (unsigned __int64 *)v5;
              v55 = (unsigned __int64 *)(v5 - v54 * 8);
            }
            v56 = 0;
            if ( v55 != v95 )
            {
              v83 = v5;
              v57 = a4;
              v58 = v55;
              while ( 1 )
              {
                v67 = *(unsigned __int8 *)(*(_QWORD *)(*v58 + 8) + 8LL);
                if ( (_BYTE)v67 == 2 )
                  break;
                if ( (_BYTE)v67 == 3 )
                {
                  v68 = 2;
                  goto LABEL_128;
                }
                if ( (_BYTE)v67 == 4 )
                {
                  v68 = 3;
                  goto LABEL_128;
                }
                if ( (unsigned int)(v67 - 17) > 1 )
                  goto LABEL_125;
                v88 = *(_QWORD *)(*v58 + 8);
                if ( sub_BCEA30(v88) )
                  goto LABEL_125;
                v59 = (__int64 *)sub_2491640(v10, *(_QWORD *)(v88 + 24));
                if ( !v59 )
                  goto LABEL_125;
                BYTE4(v102) = *(_BYTE *)(v88 + 8) == 18;
                LODWORD(v102) = *(_DWORD *)(v88 + 32);
                v60 = sub_BCE1B0(v59, v102);
LABEL_117:
                if ( v60 )
                {
                  v63 = (unsigned __int8 *)*v58;
                  if ( *(_BYTE *)*v58 <= 0x15u )
                  {
                    v64 = (unsigned __int8 *)sub_2492FB0((_QWORD **)v57, v63);
LABEL_120:
                    if ( v91 )
                      v64 = (unsigned __int8 *)sub_2495170(a1, *v58, v64, (__int64)&v112, 0, (v56 << 32) | 2);
                    v65 = (unsigned int)v110;
                    v66 = (unsigned int)v110 + 1LL;
                    if ( v66 > HIDWORD(v110) )
                    {
                      v89 = v64;
                      sub_C8D5F0((__int64)&v109, v111, v66, 8u, v61, v62);
                      v65 = (unsigned int)v110;
                      v64 = v89;
                    }
                    *(_QWORD *)&v109[8 * v65] = v64;
                    LODWORD(v110) = v110 + 1;
                    goto LABEL_125;
                  }
                  v69 = *(unsigned int *)(v57 + 32);
                  v70 = *(_QWORD *)(v57 + 16);
                  if ( (_DWORD)v69 )
                  {
                    v61 = (unsigned int)(v69 - 1);
                    v71 = v61 & (((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4));
                    v72 = (unsigned __int8 **)(v70 + 16LL * v71);
                    v73 = *v72;
                    if ( v63 == *v72 )
                    {
LABEL_133:
                      v64 = v72[1];
                      goto LABEL_120;
                    }
                    v80 = 1;
                    while ( v73 != (unsigned __int8 *)-4096LL )
                    {
                      v62 = (unsigned int)(v80 + 1);
                      v71 = v61 & (v80 + v71);
                      v72 = (unsigned __int8 **)(v70 + 16LL * v71);
                      v73 = *v72;
                      if ( v63 == *v72 )
                        goto LABEL_133;
                      v80 = v62;
                    }
                  }
                  v72 = (unsigned __int8 **)(v70 + 16 * v69);
                  goto LABEL_133;
                }
LABEL_125:
                v56 = (unsigned int)(v56 + 1);
                v58 += 4;
                if ( v95 == v58 )
                {
                  v5 = v83;
                  goto LABEL_19;
                }
              }
              v68 = 1;
LABEL_128:
              v60 = (**(__int64 (__fastcall ***)(_QWORD, _QWORD))v10[v68])(v10[v68], *(_QWORD *)(a1 + 16));
              goto LABEL_117;
            }
LABEL_19:
            v17 = *(_QWORD *)(v5 - 32);
            if ( v17 && !*(_BYTE *)v17 && *(_QWORD *)(v17 + 24) == *(_QWORD *)(v5 + 80) )
            {
              if ( (*(_BYTE *)(v17 + 33) & 0x20) != 0 || sub_981210(*a3, v17, (unsigned int *)v107) )
              {
LABEL_35:
                if ( v109 != v111 )
                  _libc_free((unsigned __int64)v109);
                nullsub_61();
                v127 = &unk_49DA100;
                nullsub_63();
                if ( v112 != v114 )
                  _libc_free((unsigned __int64)v112);
                return;
              }
              v17 = *(_QWORD *)(v5 - 32);
            }
            sub_2491700((__int64 *)&v112, v17, *(_QWORD *)(a1 + 448), 0, 0);
            v104 = 0;
            v103 = 0;
            if ( (*(_BYTE *)(v5 + 7) & 0x40) != 0 )
            {
              v18 = *(_QWORD *)(v5 - 8);
              v92 = v18 + 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF);
            }
            else
            {
              v18 = v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF);
            }
            v19 = 0;
            if ( v18 != v92 )
            {
              v85 = v10;
              v20 = v18;
              do
              {
                v94 = sub_2491640(v85, *(_QWORD *)(*(_QWORD *)v20 + 8LL));
                if ( v94 )
                {
                  v106 = 257;
                  v21 = sub_CA1930(&v103);
                  v87 = sub_2493450(
                          (__int64 *)&v112,
                          *(_QWORD *)(a1 + 456),
                          *(_QWORD *)(a1 + 464),
                          0,
                          v21,
                          (__int64)v105);
                  v90 = v19 + 1;
                  v22 = *(_QWORD *)&v109[8 * v19];
                  v108 = 257;
                  v23 = sub_BD2C40(80, unk_3F10A10);
                  v25 = (__int64)v23;
                  if ( v23 )
                    sub_B4D3C0((__int64)v23, v22, v87, 0, 0, v24, 0, 0);
                  (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v120 + 2))(
                    v120,
                    v25,
                    v107,
                    v116,
                    v117);
                  v26 = (__int64)v112;
                  v27 = &v112[16 * (unsigned int)v113];
                  if ( v112 != v27 )
                  {
                    do
                    {
                      v28 = *(_QWORD *)(v26 + 8);
                      v29 = *(_DWORD *)v26;
                      v26 += 16;
                      sub_B99FD0(v25, v29, v28);
                    }
                    while ( v27 != (_BYTE *)v26 );
                  }
                  v107[0] = sub_9208B0(*(_QWORD *)a1, v94);
                  v31 = (unsigned __int64)(v107[0] + 7LL) >> 3;
                  v107[1] = v30;
                  v103 += v31;
                  if ( v31 )
                    v104 = v30;
                  v19 = v90;
                }
                v20 += 32;
              }
              while ( v92 != v20 );
            }
            goto LABEL_35;
          }
LABEL_141:
          if ( (unsigned int)v113 >= (unsigned __int64)HIDWORD(v113) )
          {
            v81 = (unsigned int)v113 + 1LL;
            if ( HIDWORD(v113) < v81 )
            {
              v96 = (__int64)v109;
              sub_C8D5F0((__int64)&v112, v114, v81, 0x10u, (__int64)v109, v48);
              v49 = v96;
              v52 = &v112[16 * (unsigned int)v113];
            }
            *v52 = 0;
            v52[1] = v49;
            v49 = (__int64)v109;
            LODWORD(v113) = v113 + 1;
          }
          else
          {
            if ( v52 )
            {
              *(_DWORD *)v52 = 0;
              v52[1] = v49;
              v51 = v113;
              v49 = (__int64)v109;
            }
            LODWORD(v113) = v51 + 1;
          }
        }
        else
        {
          sub_93FB40((__int64)&v112, 0);
          v49 = (__int64)v109;
        }
        if ( !v49 )
          goto LABEL_103;
        goto LABEL_102;
      }
      v7 += 32;
      break;
    case 1LL:
      v10 = (_QWORD *)(a1 + 16);
      goto LABEL_70;
    default:
      return;
  }
  if ( sub_2491640(v10, *(_QWORD *)(*(_QWORD *)v7 + 8LL)) )
    goto LABEL_94;
  v7 += 32;
LABEL_70:
  if ( sub_2491640(v10, *(_QWORD *)(*(_QWORD *)v7 + 8LL)) )
    goto LABEL_94;
}
