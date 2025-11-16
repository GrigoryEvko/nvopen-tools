// Function: sub_2F3CC20
// Address: 0x2f3cc20
//
__int64 __fastcall sub_2F3CC20(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // r13
  __int64 v4; // r14
  size_t v7; // rax
  _BYTE *v8; // rdx
  _BYTE *v9; // rdi
  int v10; // eax
  unsigned int v11; // r12d
  __int64 v12; // rdx
  __int64 v13; // r13
  unsigned __int8 *v14; // r12
  __int64 v15; // rax
  __int64 v16; // rbx
  __int64 v17; // rax
  unsigned __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rsi
  __int64 v21; // r14
  unsigned __int64 v22; // rax
  int v23; // ecx
  unsigned __int64 *v24; // rdx
  int v25; // edx
  __int64 v26; // r15
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rbx
  __int64 v30; // rbx
  __int64 v31; // r14
  __int64 v32; // rbx
  __int64 v33; // rax
  unsigned int *v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rbx
  __int64 v41; // rax
  int v42; // ebx
  __int64 v43; // rax
  __int64 v44; // rdx
  int v45; // edx
  __int64 *v46; // rax
  unsigned __int8 *v47; // r14
  int v48; // ecx
  __int64 v49; // rdx
  unsigned __int8 *v50; // rbx
  __int64 v51; // rdx
  __int64 v52; // r15
  __int64 v53; // rbx
  __int64 *v54; // r15
  unsigned __int64 *v55; // rsi
  unsigned __int64 *v56; // rdx
  int v57; // ecx
  __int64 v58; // rax
  unsigned __int8 *v59; // rax
  __int64 v60; // r13
  int v61; // ecx
  unsigned __int64 v62; // r15
  unsigned __int64 *v63; // rbx
  __int64 v64; // rdx
  unsigned int v65; // esi
  const char *v66; // rax
  const char *v67; // rdx
  unsigned int v68; // eax
  int v69; // r14d
  unsigned __int64 *v70; // rbx
  unsigned __int64 *v71; // r12
  unsigned __int64 v72; // rdi
  __int64 *v73; // rax
  int v74; // r15d
  unsigned __int64 v75; // rbx
  __int64 v76; // [rsp+8h] [rbp-258h]
  __int64 v77; // [rsp+10h] [rbp-250h]
  __int64 v78; // [rsp+30h] [rbp-230h]
  unsigned int v79; // [rsp+40h] [rbp-220h]
  unsigned int v80; // [rsp+44h] [rbp-21Ch]
  __int64 v81; // [rsp+50h] [rbp-210h]
  unsigned int v83; // [rsp+68h] [rbp-1F8h]
  unsigned __int8 *v84; // [rsp+68h] [rbp-1F8h]
  __int64 v85; // [rsp+80h] [rbp-1E0h]
  __int64 v86; // [rsp+88h] [rbp-1D8h]
  _DWORD v87[8]; // [rsp+A0h] [rbp-1C0h] BYREF
  __int16 v88; // [rsp+C0h] [rbp-1A0h]
  const char *v89[4]; // [rsp+D0h] [rbp-190h] BYREF
  __int16 v90; // [rsp+F0h] [rbp-170h]
  unsigned __int64 *v91; // [rsp+100h] [rbp-160h] BYREF
  __int64 v92; // [rsp+108h] [rbp-158h]
  _BYTE v93[64]; // [rsp+110h] [rbp-150h] BYREF
  __int64 *v94; // [rsp+150h] [rbp-110h] BYREF
  __int64 v95; // [rsp+158h] [rbp-108h]
  _BYTE v96[64]; // [rsp+160h] [rbp-100h] BYREF
  unsigned __int64 *v97; // [rsp+1A0h] [rbp-C0h] BYREF
  __int64 v98; // [rsp+1A8h] [rbp-B8h]
  _BYTE v99[32]; // [rsp+1B0h] [rbp-B0h] BYREF
  __int64 v100; // [rsp+1D0h] [rbp-90h]
  unsigned __int8 *v101; // [rsp+1D8h] [rbp-88h]
  __int64 v102; // [rsp+1E0h] [rbp-80h]
  __int64 v103; // [rsp+1E8h] [rbp-78h]
  void **v104; // [rsp+1F0h] [rbp-70h]
  void **v105; // [rsp+1F8h] [rbp-68h]
  __int64 v106; // [rsp+200h] [rbp-60h]
  int v107; // [rsp+208h] [rbp-58h]
  __int16 v108; // [rsp+20Ch] [rbp-54h]
  char v109; // [rsp+20Eh] [rbp-52h]
  __int64 v110; // [rsp+210h] [rbp-50h]
  __int64 v111; // [rsp+218h] [rbp-48h]
  void *v112; // [rsp+220h] [rbp-40h] BYREF
  void *v113; // [rsp+228h] [rbp-38h] BYREF

  if ( !*(_QWORD *)(a1 + 16) )
    return 0;
  v3 = *(_QWORD *)(a1 + 40);
  v4 = *(_QWORD *)(a1 + 24);
  v7 = strlen((const char *)a2);
  v81 = sub_BA8CA0(v3, a2, v7, v4);
  v9 = v8;
  v85 = (__int64)v8;
  if ( !*v8 )
  {
    v10 = *(_BYTE *)(a1 + 32) & 0xF;
    v8 = (_BYTE *)(*(_BYTE *)(a1 + 32) & 0xF);
    if ( (unsigned int)(v10 - 7) > 1 )
    {
      a2 = v85;
      *(_BYTE *)(v85 + 32) = (unsigned __int8)v8 | *(_BYTE *)(v85 + 32) & 0xF0;
    }
    else
    {
      a2 = *(_BYTE *)(a1 + 32) & 0xF;
      *((_WORD *)v9 + 16) = a2 | *((_WORD *)v9 + 16) & 0xFCC0;
      if ( v10 == 7 )
        goto LABEL_5;
    }
    if ( v10 != 8 && ((*(_BYTE *)(v85 + 32) & 0x30) == 0 || (_BYTE)v8 == 9) )
    {
LABEL_6:
      if ( a3 )
      {
        if ( (((_BYTE)v8 + 14) & 0xFu) > 3 )
        {
          v8 = (_BYTE *)(((_BYTE)v8 + 7) & 0xF);
          if ( (unsigned __int8)v8 > 1u )
          {
            a2 = 42;
            sub_B2CD30(v85, 42);
          }
        }
      }
      goto LABEL_8;
    }
LABEL_5:
    *(_BYTE *)(v85 + 33) |= 0x40u;
    goto LABEL_6;
  }
LABEL_8:
  v11 = sub_3108960(a1, a2, v8);
  v79 = 1;
  if ( !(unsigned __int8)sub_3108D30(v11, a2, v12) )
    v79 = (unsigned __int8)sub_3108D60(v11) != 0 ? 3 : 0;
  v86 = *(_QWORD *)(a1 + 16);
  while ( v86 )
  {
    v13 = v86;
    v14 = *(unsigned __int8 **)(v86 + 24);
    v86 = *(_QWORD *)(v86 + 8);
    v15 = *((_QWORD *)v14 - 4);
    if ( !v15 || *(_BYTE *)v15 || *(_QWORD *)(v15 + 24) != *((_QWORD *)v14 + 10) || a1 != v15 )
    {
      if ( (v14[7] & 0x80u) != 0 )
      {
        v27 = sub_BD2BC0((__int64)v14);
        v29 = v27 + v28;
        if ( (v14[7] & 0x80u) != 0 )
          v29 -= sub_BD2BC0((__int64)v14);
        v30 = v29 >> 4;
        if ( (_DWORD)v30 )
        {
          v31 = 0;
          v32 = 16LL * (unsigned int)v30;
          while ( 1 )
          {
            v33 = 0;
            if ( (v14[7] & 0x80u) != 0 )
              v33 = sub_BD2BC0((__int64)v14);
            v34 = (unsigned int *)(v31 + v33);
            if ( *(_DWORD *)(*(_QWORD *)v34 + 8LL) == 6 )
              break;
            v31 += 16;
            if ( v32 == v31 )
              goto LABEL_38;
          }
          sub_3108960(
            *(_QWORD *)&v14[32 * (v34[2] - (unsigned __int64)(*((_DWORD *)v14 + 1) & 0x7FFFFFF))],
            a2,
            *((_DWORD *)v14 + 1) & 0x7FFFFFF);
        }
      }
LABEL_38:
      if ( *(_QWORD *)v13 )
      {
        v35 = *(_QWORD *)(v13 + 8);
        **(_QWORD **)(v13 + 16) = v35;
        if ( v35 )
          *(_QWORD *)(v35 + 16) = *(_QWORD *)(v13 + 16);
      }
      *(_QWORD *)v13 = v85;
      v36 = *(_QWORD *)(v85 + 16);
      *(_QWORD *)(v13 + 8) = v36;
      if ( v36 )
        *(_QWORD *)(v36 + 16) = v13 + 8;
      *(_QWORD *)(v13 + 16) = v85 + 16;
      *(_QWORD *)(v85 + 16) = v13;
      continue;
    }
    v16 = *((_QWORD *)v14 + 5);
    v17 = sub_AA48A0(v16);
    v109 = 7;
    v103 = v17;
    v104 = &v112;
    v105 = &v113;
    v97 = (unsigned __int64 *)v99;
    v112 = &unk_49DA100;
    v100 = v16;
    v98 = 0x200000000LL;
    v106 = 0;
    v107 = 0;
    v108 = 512;
    v110 = 0;
    v111 = 0;
    v113 = &unk_49DA0B0;
    v101 = v14 + 24;
    LOWORD(v102) = 0;
    if ( v14 + 24 != (unsigned __int8 *)(v16 + 48) )
    {
      v20 = *(_QWORD *)sub_B46C60((__int64)v14);
      v94 = (__int64 *)v20;
      if ( v20 && (sub_B96E90((__int64)&v94, v20, 1), (v21 = (__int64)v94) != 0) )
      {
        v18 = (unsigned int)v98;
        v22 = (unsigned __int64)v97;
        v23 = v98;
        v24 = &v97[2 * (unsigned int)v98];
        if ( v97 != v24 )
        {
          while ( *(_DWORD *)v22 )
          {
            v22 += 16LL;
            if ( v24 == (unsigned __int64 *)v22 )
              goto LABEL_98;
          }
          *(_QWORD *)(v22 + 8) = v94;
LABEL_23:
          sub_B91220((__int64)&v94, v21);
          goto LABEL_24;
        }
LABEL_98:
        if ( (unsigned int)v98 >= (unsigned __int64)HIDWORD(v98) )
        {
          v18 = (unsigned int)v98 + 1LL;
          v75 = v76 & 0xFFFFFFFF00000000LL;
          v76 &= 0xFFFFFFFF00000000LL;
          if ( HIDWORD(v98) < v18 )
          {
            sub_C8D5F0((__int64)&v97, v99, v18, 0x10u, v18, v19);
            v24 = &v97[2 * (unsigned int)v98];
          }
          *v24 = v75;
          v24[1] = v21;
          v21 = (__int64)v94;
          LODWORD(v98) = v98 + 1;
        }
        else
        {
          if ( v24 )
          {
            *(_DWORD *)v24 = 0;
            v24[1] = v21;
            v23 = v98;
            v21 = (__int64)v94;
          }
          LODWORD(v98) = v23 + 1;
        }
      }
      else
      {
        sub_93FB40((__int64)&v97, 0);
        v21 = (__int64)v94;
      }
      if ( v21 )
        goto LABEL_23;
    }
LABEL_24:
    v25 = *v14;
    if ( v25 == 40 )
    {
      v26 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)v14);
    }
    else
    {
      v26 = -32;
      if ( v25 != 85 )
      {
        if ( v25 != 34 )
          BUG();
        v26 = -96;
      }
    }
    if ( (v14[7] & 0x80u) != 0 )
    {
      v38 = sub_BD2BC0((__int64)v14);
      v40 = v38 + v39;
      v41 = 0;
      if ( (v14[7] & 0x80u) != 0 )
        v41 = sub_BD2BC0((__int64)v14);
      if ( (unsigned int)((v40 - v41) >> 4) )
      {
        if ( (v14[7] & 0x80u) == 0 )
          BUG();
        v42 = *(_DWORD *)(sub_BD2BC0((__int64)v14) + 8);
        if ( (v14[7] & 0x80u) == 0 )
          BUG();
        v43 = sub_BD2BC0((__int64)v14);
        v26 -= 32LL * (unsigned int)(*(_DWORD *)(v43 + v44 - 4) - v42);
      }
    }
    v45 = *((_DWORD *)v14 + 1);
    v46 = (__int64 *)v96;
    v47 = &v14[v26];
    v95 = 0x800000000LL;
    v48 = 0;
    v49 = 32LL * (v45 & 0x7FFFFFF);
    v94 = (__int64 *)v96;
    v50 = &v14[-v49];
    v51 = v26 + v49;
    v52 = v51 >> 5;
    if ( (unsigned __int64)v51 > 0x100 )
    {
      sub_C8D5F0((__int64)&v94, v96, v51 >> 5, 8u, v18, v19);
      v48 = v95;
      v46 = &v94[(unsigned int)v95];
    }
    if ( v50 != v47 )
    {
      do
      {
        if ( v46 )
          *v46 = *(_QWORD *)v50;
        v50 += 32;
        ++v46;
      }
      while ( v47 != v50 );
      v48 = v95;
    }
    v91 = (unsigned __int64 *)v93;
    v92 = 0x100000000LL;
    LODWORD(v95) = v52 + v48;
    sub_B56970((__int64)v14, (__int64)&v91);
    v53 = (unsigned int)v95;
    v88 = 257;
    v54 = v94;
    v90 = 257;
    v55 = &v91[7 * (unsigned int)v92];
    if ( v91 == v55 )
    {
      v57 = 0;
    }
    else
    {
      v56 = v91;
      v57 = 0;
      do
      {
        v58 = v56[5] - v56[4];
        v56 += 7;
        v57 += v58 >> 3;
      }
      while ( v55 != v56 );
    }
    v77 = (unsigned int)v92;
    LOBYTE(v47) = 16 * (_DWORD)v92 != 0;
    v83 = v95 + v57 + 1;
    v78 = (__int64)v91;
    v59 = (unsigned __int8 *)sub_BD2CC0(88, ((unsigned __int64)(unsigned int)(16 * v92) << 32) | v83);
    v60 = (__int64)v59;
    if ( v59 )
    {
      v61 = v83 & 0x7FFFFFF | ((_DWORD)v47 << 28);
      v84 = v59;
      v80 = v80 & 0xE0000000 | v61;
      sub_B44260((__int64)v59, **(_QWORD **)(v81 + 16), 56, v80, 0, 0);
      *(_QWORD *)(v60 + 72) = 0;
      sub_B4A290(v60, v81, v85, v54, v53, (__int64)v89, v78, v77);
    }
    else
    {
      v84 = 0;
    }
    if ( (_BYTE)v108 )
    {
      v73 = (__int64 *)sub_BD5C60((__int64)v84);
      *(_QWORD *)(v60 + 72) = sub_A7A090((__int64 *)(v60 + 72), v73, -1, 72);
      if ( (unsigned __int8)sub_920620((__int64)v84) )
      {
LABEL_92:
        v74 = v107;
        if ( v106 )
          sub_B99FD0(v60, 3u, v106);
        sub_B45150(v60, v74);
      }
    }
    else if ( (unsigned __int8)sub_920620((__int64)v84) )
    {
      goto LABEL_92;
    }
    (*((void (__fastcall **)(void **, __int64, _DWORD *, unsigned __int8 *, __int64))*v105 + 2))(
      v105,
      v60,
      v87,
      v101,
      v102);
    v62 = (unsigned __int64)v97;
    v63 = &v97[2 * (unsigned int)v98];
    if ( v97 != v63 )
    {
      do
      {
        v64 = *(_QWORD *)(v62 + 8);
        v65 = *(_DWORD *)v62;
        v62 += 16LL;
        sub_B99FD0(v60, v65, v64);
      }
      while ( v63 != (unsigned __int64 *)v62 );
    }
    v66 = sub_BD5D20((__int64)v14);
    v90 = 261;
    v89[0] = v66;
    v89[1] = v67;
    sub_BD6B50(v84, v89);
    a2 = 52;
    v68 = *((_WORD *)v14 + 1) & 3;
    if ( v68 < v79 )
      LOWORD(v68) = v79;
    *(_WORD *)(v60 + 2) = *(_WORD *)(v60 + 2) & 0xFFFC | v68;
    v89[0] = *(const char **)(a1 + 120);
    if ( (unsigned __int8)sub_A74390((__int64 *)v89, 52, v87) )
    {
      v69 = v87[0];
      if ( v87[0] )
      {
        a2 = sub_BD5C60((__int64)v84);
        *(_QWORD *)(v60 + 72) = sub_A7A090((__int64 *)(v60 + 72), (__int64 *)a2, v69, 52);
      }
    }
    if ( *((_QWORD *)v14 + 2) )
    {
      a2 = v60;
      sub_BD84D0((__int64)v14, v60);
    }
    sub_B43D60(v14);
    v70 = v91;
    v71 = &v91[7 * (unsigned int)v92];
    if ( v91 != v71 )
    {
      do
      {
        v72 = *(v71 - 3);
        v71 -= 7;
        if ( v72 )
        {
          a2 = v71[6] - v72;
          j_j___libc_free_0(v72);
        }
        if ( (unsigned __int64 *)*v71 != v71 + 2 )
        {
          a2 = v71[2] + 1;
          j_j___libc_free_0(*v71);
        }
      }
      while ( v70 != v71 );
      v71 = v91;
    }
    if ( v71 != (unsigned __int64 *)v93 )
      _libc_free((unsigned __int64)v71);
    if ( v94 != (__int64 *)v96 )
      _libc_free((unsigned __int64)v94);
    nullsub_61();
    v112 = &unk_49DA100;
    nullsub_63();
    if ( v97 != (unsigned __int64 *)v99 )
      _libc_free((unsigned __int64)v97);
  }
  return 1;
}
