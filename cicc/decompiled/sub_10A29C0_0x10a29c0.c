// Function: sub_10A29C0
// Address: 0x10a29c0
//
__int64 __fastcall sub_10A29C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 v5; // r15
  _BYTE *v7; // r8
  __int64 v8; // rax
  _QWORD *v9; // r10
  _BYTE *v11; // rax
  char v12; // dl
  bool v13; // al
  __int64 v14; // r14
  char v15; // dl
  unsigned __int8 v16; // cl
  __int64 v17; // r11
  _BYTE *v18; // rbx
  __int64 v19; // rax
  _BYTE *v20; // rbx
  __int64 v21; // r13
  unsigned __int8 *v22; // r14
  unsigned __int64 v23; // rax
  __int64 v24; // rax
  char v25; // al
  unsigned int v26; // edx
  __int64 v27; // rsi
  __int64 v28; // r8
  unsigned __int64 v29; // rax
  unsigned int v30; // edx
  int v31; // eax
  char **v32; // r8
  int v33; // r14d
  unsigned int v34; // edx
  _BYTE **v35; // r8
  int v36; // eax
  unsigned __int8 *v37; // r14
  char v38; // al
  char v39; // r8
  unsigned __int8 v40; // al
  int v41; // edx
  _QWORD *v42; // rax
  _BYTE *v43; // rax
  __int64 v44; // r12
  _QWORD *v45; // rax
  __int64 v46; // rdx
  int v47; // r8d
  unsigned int *v48; // rax
  __int64 v49; // rdx
  int v50; // r15d
  unsigned int *v51; // rbx
  __int64 v52; // r12
  __int64 v53; // rdx
  unsigned int v54; // esi
  __int16 v55; // ax
  char v56; // al
  __int64 v57; // rdx
  char v58; // al
  __int64 v59; // rbx
  _BYTE *v60; // r10
  __int64 v61; // rax
  _QWORD *v62; // rax
  unsigned int *v63; // rbx
  __int64 v64; // r12
  __int64 v65; // rdx
  unsigned int v66; // esi
  _QWORD *v67; // rax
  _QWORD *v68; // r10
  unsigned int *v69; // rbx
  __int64 v70; // rdx
  unsigned int v71; // esi
  __int16 v72; // ax
  unsigned __int64 v73; // [rsp+10h] [rbp-D0h]
  unsigned int v74; // [rsp+18h] [rbp-C8h]
  _BYTE **v75; // [rsp+18h] [rbp-C8h]
  _BYTE *v76; // [rsp+20h] [rbp-C0h]
  __int64 v77; // [rsp+20h] [rbp-C0h]
  unsigned int v78; // [rsp+20h] [rbp-C0h]
  __int64 v79; // [rsp+20h] [rbp-C0h]
  __int64 v80; // [rsp+20h] [rbp-C0h]
  _QWORD *v81; // [rsp+28h] [rbp-B8h]
  __int64 v82; // [rsp+28h] [rbp-B8h]
  __int64 v83; // [rsp+28h] [rbp-B8h]
  _BYTE *v84; // [rsp+28h] [rbp-B8h]
  _QWORD *v85; // [rsp+28h] [rbp-B8h]
  _QWORD *v86; // [rsp+28h] [rbp-B8h]
  _QWORD *v87; // [rsp+28h] [rbp-B8h]
  _QWORD *v88; // [rsp+28h] [rbp-B8h]
  int v89; // [rsp+28h] [rbp-B8h]
  unsigned int *v90; // [rsp+28h] [rbp-B8h]
  _BYTE *v91; // [rsp+28h] [rbp-B8h]
  _QWORD *v92; // [rsp+28h] [rbp-B8h]
  _BYTE *v93; // [rsp+28h] [rbp-B8h]
  _BYTE *v94; // [rsp+28h] [rbp-B8h]
  _BYTE *v95; // [rsp+30h] [rbp-B0h] BYREF
  __int64 *v96; // [rsp+38h] [rbp-A8h] BYREF
  unsigned __int64 v97; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v98; // [rsp+48h] [rbp-98h]
  unsigned __int64 v99; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v100; // [rsp+58h] [rbp-88h]
  __int16 v101; // [rsp+70h] [rbp-70h]
  _BYTE **v102; // [rsp+80h] [rbp-60h] BYREF
  __int64 **v103; // [rsp+88h] [rbp-58h] BYREF
  char v104; // [rsp+90h] [rbp-50h]
  _QWORD *v105; // [rsp+98h] [rbp-48h]
  __int64 **v106; // [rsp+A0h] [rbp-40h] BYREF
  char v107; // [rsp+A8h] [rbp-38h]

  v3 = *(_QWORD *)(a1 - 32);
  if ( *(_BYTE *)v3 > 0x15u )
    return 0;
  v4 = *(_QWORD *)(a1 - 64);
  v5 = *(_QWORD *)(a1 + 8);
  v7 = (_BYTE *)(v3 + 24);
  if ( *(_BYTE *)v3 != 17 )
  {
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v3 + 8) + 8LL) - 17 > 1 )
      goto LABEL_4;
    v11 = sub_AD7630(*(_QWORD *)(a1 - 32), 0, a3);
    if ( !v11 || *v11 != 17 )
      goto LABEL_4;
    v7 = v11 + 24;
  }
  v104 = 0;
  v102 = &v95;
  v103 = &v96;
  v105 = &v95;
  v106 = &v96;
  v107 = 0;
  if ( *(_BYTE *)v4 != 68 )
  {
LABEL_4:
    v8 = *(_QWORD *)(v4 + 16);
    goto LABEL_5;
  }
  v22 = *(unsigned __int8 **)(v4 - 32);
  v23 = *v22;
  if ( (unsigned __int8)v23 <= 0x1Cu )
  {
    if ( (_BYTE)v23 != 5 )
      goto LABEL_4;
    v55 = *((_WORD *)v22 + 1);
    if ( (v55 & 0xFFF7) != 0x11 && (v55 & 0xFFFD) != 0xD )
      goto LABEL_4;
    if ( v55 != 13 )
      goto LABEL_4;
  }
  else
  {
    if ( (unsigned __int8)v23 > 0x36u )
      goto LABEL_30;
    v57 = 0x40540000000000LL;
    if ( !_bittest64(&v57, v23) || (_BYTE)v23 != 42 )
      goto LABEL_4;
  }
  v91 = v7;
  if ( (v22[1] & 2) == 0 || !*((_QWORD *)v22 - 8) )
    goto LABEL_4;
  v95 = (_BYTE *)*((_QWORD *)v22 - 8);
  v56 = sub_991580((__int64)&v103, *((_QWORD *)v22 - 4));
  v7 = v91;
  if ( v56 )
    goto LABEL_34;
  LOBYTE(v23) = *v22;
LABEL_30:
  v84 = v7;
  if ( (_BYTE)v23 != 58 )
    goto LABEL_4;
  if ( (v22[1] & 2) == 0 )
    goto LABEL_4;
  v24 = *((_QWORD *)v22 - 8);
  if ( !v24 )
    goto LABEL_4;
  *v105 = v24;
  v25 = sub_991580((__int64)&v106, *((_QWORD *)v22 - 4));
  v7 = v84;
  if ( !v25 )
    goto LABEL_4;
LABEL_34:
  v26 = *((_DWORD *)v7 + 2);
  v27 = *(_QWORD *)v7;
  if ( v26 > 0x40 )
    v27 = *(_QWORD *)(v27 + 8LL * ((v26 - 1) >> 6));
  if ( (v27 & (1LL << ((unsigned __int8)v26 - 1))) == 0 )
    goto LABEL_4;
  v76 = v7;
  sub_C44830((__int64)&v97, v96, v26);
  v28 = (__int64)v76;
  if ( v98 > 0x40 )
  {
    sub_C43D10((__int64)&v97);
    v28 = (__int64)v76;
  }
  else
  {
    v29 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v98) & ~v97;
    if ( !v98 )
      v29 = 0;
    v97 = v29;
  }
  v77 = v28;
  sub_C46250((__int64)&v97);
  v30 = v98;
  v98 = 0;
  v100 = v30;
  v74 = v30;
  v99 = v97;
  v73 = v97;
  v31 = sub_C4C880(v77, (__int64)&v99);
  v32 = (char **)v77;
  v33 = v31;
  if ( v74 > 0x40 )
  {
    if ( v73 )
    {
      j_j___libc_free_0_0(v73);
      v32 = (char **)v77;
      if ( v98 > 0x40 )
      {
        if ( v97 )
        {
          j_j___libc_free_0_0(v97);
          v32 = (char **)v77;
        }
      }
    }
  }
  if ( v33 < 0 )
    goto LABEL_4;
  sub_C44740((__int64)&v102, v32, *((_DWORD *)v96 + 2));
  sub_C45EE0((__int64)&v102, v96);
  v34 = (unsigned int)v103;
  v35 = v102;
  v98 = (unsigned int)v103;
  v97 = (unsigned __int64)v102;
  if ( (unsigned int)v103 <= 0x40 )
  {
    if ( v102 )
      goto LABEL_49;
LABEL_64:
    LOWORD(v106) = 257;
    v42 = sub_BD2C40(72, unk_3F10A14);
    v9 = v42;
    if ( v42 )
    {
      v86 = v42;
      sub_B515B0((__int64)v42, (__int64)v95, v5, (__int64)&v102, 0, 0);
      v9 = v86;
    }
    goto LABEL_66;
  }
  v75 = v102;
  v78 = (unsigned int)v103;
  v36 = sub_C444A0((__int64)&v97);
  v34 = v78;
  v35 = v75;
  if ( v78 == v36 )
    goto LABEL_64;
LABEL_49:
  v8 = *(_QWORD *)(v4 + 16);
  if ( v8 && !*(_QWORD *)(v8 + 8) )
  {
    v101 = 257;
    v43 = (_BYTE *)sub_AD8D80(*((_QWORD *)v95 + 1), (__int64)&v97);
    v44 = sub_929C50((unsigned int **)a2, v95, v43, (__int64)&v99, 1u, 0);
    LOWORD(v106) = 257;
    v45 = sub_BD2C40(72, unk_3F10A14);
    v9 = v45;
    if ( v45 )
    {
      v88 = v45;
      sub_B515B0((__int64)v45, v44, v5, (__int64)&v102, 0, 0);
      v9 = v88;
    }
LABEL_66:
    if ( v98 > 0x40 && v97 )
    {
      v87 = v9;
      j_j___libc_free_0_0(v97);
      return (__int64)v87;
    }
    return (__int64)v9;
  }
  if ( v34 > 0x40 && v35 )
  {
    j_j___libc_free_0_0(v35);
    v8 = *(_QWORD *)(v4 + 16);
  }
LABEL_5:
  if ( !v8 )
    return 0;
  v9 = *(_QWORD **)(v8 + 8);
  if ( v9 )
    return 0;
  v12 = *(_BYTE *)v4;
  if ( *(_BYTE *)v4 != 69 )
    goto LABEL_14;
  v37 = *(unsigned __int8 **)(v4 - 32);
  if ( !v37 )
    BUG();
  v85 = *(_QWORD **)(v8 + 8);
  v38 = sub_987880(*(unsigned __int8 **)(v4 - 32));
  v9 = v85;
  v39 = v38;
  v40 = *v37;
  if ( !v39
    || (v40 <= 0x1Cu ? (v41 = *((unsigned __int16 *)v37 + 1)) : (v41 = v40 - 29),
        v41 != 13 || (v37[1] & 4) == 0 || !*((_QWORD *)v37 - 8)) )
  {
    if ( v40 != 58 )
      return (__int64)v9;
    goto LABEL_130;
  }
  v95 = (_BYTE *)*((_QWORD *)v37 - 8);
  v17 = *((_QWORD *)v37 - 4);
  if ( *(_BYTE *)v17 <= 0x15u )
    goto LABEL_23;
  if ( *v37 == 58 )
  {
LABEL_130:
    if ( (v37[1] & 2) != 0 )
    {
      if ( *((_QWORD *)v37 - 8) )
      {
        v95 = (_BYTE *)*((_QWORD *)v37 - 8);
        v17 = *((_QWORD *)v37 - 4);
        if ( *(_BYTE *)v17 <= 0x15u )
        {
LABEL_23:
          v101 = 257;
          if ( v5 == *(_QWORD *)(v17 + 8) )
          {
            v18 = (_BYTE *)v17;
          }
          else
          {
            v82 = v17;
            v18 = (_BYTE *)(*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a2 + 80) + 120LL))(
                             *(_QWORD *)(a2 + 80),
                             40,
                             v17,
                             v5);
            if ( !v18 )
            {
              LOWORD(v106) = 257;
              v18 = (_BYTE *)sub_B51D30(40, v82, v5, (__int64)&v102, 0, 0);
              if ( (unsigned __int8)sub_920620((__int64)v18) )
              {
                v46 = *(_QWORD *)(a2 + 96);
                v47 = *(_DWORD *)(a2 + 104);
                if ( v46 )
                {
                  v89 = *(_DWORD *)(a2 + 104);
                  sub_B99FD0((__int64)v18, 3u, v46);
                  v47 = v89;
                }
                sub_B45150((__int64)v18, v47);
              }
              (*(void (__fastcall **)(_QWORD, _BYTE *, unsigned __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
                *(_QWORD *)(a2 + 88),
                v18,
                &v99,
                *(_QWORD *)(a2 + 56),
                *(_QWORD *)(a2 + 64));
              v48 = *(unsigned int **)a2;
              v79 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
              if ( *(_QWORD *)a2 != v79 )
              {
                do
                {
                  v90 = v48;
                  sub_B99FD0((__int64)v18, *v48, *((_QWORD *)v48 + 1));
                  v48 = v90 + 4;
                }
                while ( (unsigned int *)v79 != v90 + 4 );
              }
            }
          }
          LOWORD(v106) = 257;
          v19 = sub_929C50((unsigned int **)a2, v18, (_BYTE *)v3, (__int64)&v102, 0, 0);
          v20 = v95;
          v83 = v19;
          v101 = 257;
          if ( v5 != *((_QWORD *)v95 + 1) )
          {
            v21 = (*(__int64 (__fastcall **)(_QWORD, __int64, _BYTE *, __int64))(**(_QWORD **)(a2 + 80) + 120LL))(
                    *(_QWORD *)(a2 + 80),
                    40,
                    v95,
                    v5);
            if ( !v21 )
            {
              LOWORD(v106) = 257;
              v21 = sub_B51D30(40, (__int64)v20, v5, (__int64)&v102, 0, 0);
              if ( (unsigned __int8)sub_920620(v21) )
              {
                v49 = *(_QWORD *)(a2 + 96);
                v50 = *(_DWORD *)(a2 + 104);
                if ( v49 )
                  sub_B99FD0(v21, 3u, v49);
                sub_B45150(v21, v50);
              }
              (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
                *(_QWORD *)(a2 + 88),
                v21,
                &v99,
                *(_QWORD *)(a2 + 56),
                *(_QWORD *)(a2 + 64));
              v51 = *(unsigned int **)a2;
              v52 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
              while ( (unsigned int *)v52 != v51 )
              {
                v53 = *((_QWORD *)v51 + 1);
                v54 = *v51;
                v51 += 4;
                sub_B99FD0(v21, v54, v53);
              }
            }
            goto LABEL_27;
          }
LABEL_69:
          v21 = (__int64)v20;
LABEL_27:
          LOWORD(v106) = 257;
          return sub_B504D0(13, v21, v83, (__int64)&v102, 0, 0);
        }
      }
    }
  }
  v12 = *(_BYTE *)v4;
LABEL_14:
  if ( v12 != 68 )
    return (__int64)v9;
  v81 = v9;
  v13 = sub_B44910(v4);
  v14 = *(_QWORD *)(v4 - 32);
  v9 = v81;
  v15 = 68;
  if ( v13 )
  {
    v16 = *(_BYTE *)v14;
    if ( *(_BYTE *)v14 <= 0x1Cu )
    {
      if ( v16 != 5 )
        goto LABEL_107;
      v72 = *(_WORD *)(v14 + 2);
      if ( (v72 & 0xFFFD) != 0xD && (v72 & 0xFFF7) != 0x11 )
        goto LABEL_107;
      if ( v72 != 13 )
        goto LABEL_107;
    }
    else
    {
      if ( v16 > 0x36u )
        goto LABEL_100;
      if ( ((0x40540000000000uLL >> v16) & 1) == 0 )
        goto LABEL_105;
      if ( v16 != 42 )
      {
LABEL_100:
        if ( *(_BYTE *)v14 == 58 && (*(_BYTE *)(v14 + 1) & 2) != 0 )
        {
          if ( *(_QWORD *)(v14 - 64) )
          {
            v95 = *(_BYTE **)(v14 - 64);
            v17 = *(_QWORD *)(v14 - 32);
            if ( *(_BYTE *)v17 <= 0x15u )
              goto LABEL_23;
          }
        }
        v15 = *(_BYTE *)v4;
        goto LABEL_105;
      }
    }
    if ( (*(_BYTE *)(v14 + 1) & 4) != 0 )
    {
      if ( *(_QWORD *)(v14 - 64) )
      {
        v95 = *(_BYTE **)(v14 - 64);
        v17 = *(_QWORD *)(v14 - 32);
        if ( *(_BYTE *)v17 <= 0x15u )
          goto LABEL_23;
      }
      goto LABEL_100;
    }
LABEL_105:
    if ( v15 != 68 )
      return (__int64)v9;
    v14 = *(_QWORD *)(v4 - 32);
  }
LABEL_107:
  if ( !v14 )
    BUG();
  v58 = sub_987880((unsigned __int8 *)v14);
  v9 = v81;
  if ( v58 )
  {
    if ( *(_BYTE *)v14 <= 0x1Cu )
    {
      if ( *(_WORD *)(v14 + 2) != 13 )
        return (__int64)v9;
      goto LABEL_111;
    }
    if ( *(_BYTE *)v14 == 42 )
    {
LABEL_111:
      if ( (*(_BYTE *)(v14 + 1) & 2) == 0 || !*(_QWORD *)(v14 - 64) )
        return (__int64)v9;
      v95 = *(_BYTE **)(v14 - 64);
      v59 = *(_QWORD *)(v14 - 32);
      if ( *(_BYTE *)v59 <= 0x15u )
        goto LABEL_114;
    }
  }
  if ( *(_BYTE *)v14 == 58 && (*(_BYTE *)(v14 + 1) & 2) != 0 )
  {
    if ( *(_QWORD *)(v14 - 64) )
    {
      v95 = *(_BYTE **)(v14 - 64);
      v59 = *(_QWORD *)(v14 - 32);
      if ( *(_BYTE *)v59 <= 0x15u )
      {
LABEL_114:
        v101 = 257;
        if ( v5 == *(_QWORD *)(v59 + 8) )
        {
          v60 = (_BYTE *)v59;
        }
        else
        {
          v60 = (_BYTE *)(*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a2 + 80) + 120LL))(
                           *(_QWORD *)(a2 + 80),
                           39,
                           v59,
                           v5);
          if ( !v60 )
          {
            LOWORD(v106) = 257;
            v67 = sub_BD2C40(72, unk_3F10A14);
            v68 = v67;
            if ( v67 )
            {
              v92 = v67;
              sub_B515B0((__int64)v67, v59, v5, (__int64)&v102, 0, 0);
              v68 = v92;
            }
            v93 = v68;
            (*(void (__fastcall **)(_QWORD, _QWORD *, unsigned __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
              *(_QWORD *)(a2 + 88),
              v68,
              &v99,
              *(_QWORD *)(a2 + 56),
              *(_QWORD *)(a2 + 64));
            v60 = v93;
            v69 = *(unsigned int **)a2;
            v80 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
            if ( *(_QWORD *)a2 != v80 )
            {
              do
              {
                v70 = *((_QWORD *)v69 + 1);
                v71 = *v69;
                v94 = v60;
                v69 += 4;
                sub_B99FD0((__int64)v60, v71, v70);
                v60 = v94;
              }
              while ( (unsigned int *)v80 != v69 );
            }
          }
        }
        LOWORD(v106) = 257;
        v61 = sub_929C50((unsigned int **)a2, v60, (_BYTE *)v3, (__int64)&v102, 0, 0);
        v20 = v95;
        v83 = v61;
        v101 = 257;
        if ( v5 != *((_QWORD *)v95 + 1) )
        {
          v21 = (*(__int64 (__fastcall **)(_QWORD, __int64, _BYTE *, __int64))(**(_QWORD **)(a2 + 80) + 120LL))(
                  *(_QWORD *)(a2 + 80),
                  39,
                  v95,
                  v5);
          if ( !v21 )
          {
            LOWORD(v106) = 257;
            v62 = sub_BD2C40(72, unk_3F10A14);
            v21 = (__int64)v62;
            if ( v62 )
              sub_B515B0((__int64)v62, (__int64)v20, v5, (__int64)&v102, 0, 0);
            (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
              *(_QWORD *)(a2 + 88),
              v21,
              &v99,
              *(_QWORD *)(a2 + 56),
              *(_QWORD *)(a2 + 64));
            v63 = *(unsigned int **)a2;
            v64 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
            while ( (unsigned int *)v64 != v63 )
            {
              v65 = *((_QWORD *)v63 + 1);
              v66 = *v63;
              v63 += 4;
              sub_B99FD0(v21, v66, v65);
            }
          }
          goto LABEL_27;
        }
        goto LABEL_69;
      }
    }
  }
  return (__int64)v9;
}
