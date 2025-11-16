// Function: sub_9D5A00
// Address: 0x9d5a00
//
__int64 *__fastcall sub_9D5A00(__int64 *a1, unsigned __int64 *a2)
{
  __int64 *v2; // r13
  __int64 v3; // rcx
  __int64 *v4; // r15
  __int64 v5; // r14
  unsigned __int64 *v7; // rsi
  char v8; // dl
  int v9; // edx
  __int64 *v11; // r14
  __int64 v12; // rcx
  __int64 v13; // rax
  unsigned int v14; // r15d
  unsigned int v15; // eax
  __int64 *v16; // rdx
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rax
  __int64 v22; // rax
  unsigned int v23; // esi
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rax
  unsigned int v27; // ebx
  int v28; // r13d
  char v29; // dl
  __int64 v30; // rdx
  unsigned __int64 *v31; // rsi
  unsigned __int64 v32; // rax
  int v33; // ecx
  unsigned __int64 v34; // rax
  unsigned int v35; // edx
  unsigned int v36; // edx
  __int64 v37; // rax
  unsigned int v38; // eax
  __int64 v39; // r14
  __int64 *v40; // rsi
  __int64 v41; // rdx
  unsigned __int64 v42; // rdi
  unsigned __int64 *v43; // r9
  __int64 v44; // r10
  __int64 v45; // r8
  unsigned __int64 v46; // rsi
  unsigned __int64 v47; // rdi
  __int64 v48; // rsi
  __int64 v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rcx
  __int64 v52; // rdx
  char v53; // dl
  __int64 v54; // rsi
  __int64 v55; // rbx
  unsigned __int64 v56; // rax
  unsigned __int64 *v57; // r8
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 *i; // rdx
  __int64 v62; // r14
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rdx
  bool v66; // al
  unsigned __int64 *v67; // r13
  unsigned __int64 *v68; // rbx
  unsigned __int64 v69; // rdi
  unsigned __int64 *v70; // r12
  unsigned __int64 *v71; // rbx
  unsigned __int64 v72; // rdi
  unsigned __int64 v73; // rax
  unsigned __int16 v74; // cx
  unsigned __int64 v75; // rax
  unsigned __int16 v76; // cx
  unsigned __int64 v77; // rsi
  __int64 v78; // rdi
  unsigned __int64 *v79; // rbx
  const char *v80; // rax
  __int64 v81; // rax
  unsigned int v82; // ecx
  unsigned __int64 *v83; // rbx
  const char *v84; // rax
  unsigned __int16 v85; // [rsp+2h] [rbp-40Eh]
  unsigned __int16 v86; // [rsp+4h] [rbp-40Ch]
  unsigned __int16 v87; // [rsp+6h] [rbp-40Ah]
  char v88; // [rsp+18h] [rbp-3F8h]
  unsigned __int64 v89; // [rsp+18h] [rbp-3F8h]
  char *v90; // [rsp+18h] [rbp-3F8h]
  __int64 v91; // [rsp+20h] [rbp-3F0h]
  unsigned int v92; // [rsp+30h] [rbp-3E0h]
  unsigned int v93; // [rsp+3Ch] [rbp-3D4h]
  int v94; // [rsp+40h] [rbp-3D0h]
  unsigned __int64 *v95; // [rsp+40h] [rbp-3D0h]
  __int64 *v96; // [rsp+48h] [rbp-3C8h]
  __int64 *v97; // [rsp+50h] [rbp-3C0h]
  _QWORD *v98; // [rsp+50h] [rbp-3C0h]
  __int64 v99; // [rsp+58h] [rbp-3B8h]
  __int64 v100; // [rsp+58h] [rbp-3B8h]
  __int64 v101; // [rsp+68h] [rbp-3A8h]
  __int64 v102; // [rsp+68h] [rbp-3A8h]
  unsigned int v103; // [rsp+80h] [rbp-390h] BYREF
  unsigned int v104; // [rsp+84h] [rbp-38Ch] BYREF
  __int64 v105; // [rsp+88h] [rbp-388h] BYREF
  __int64 v106; // [rsp+90h] [rbp-380h] BYREF
  char v107; // [rsp+98h] [rbp-378h]
  __int64 v108; // [rsp+A0h] [rbp-370h] BYREF
  char v109; // [rsp+A8h] [rbp-368h]
  unsigned __int64 v110; // [rsp+B0h] [rbp-360h] BYREF
  _BYTE *v111; // [rsp+B8h] [rbp-358h]
  __int64 v112; // [rsp+C0h] [rbp-350h]
  _BYTE v113[72]; // [rsp+C8h] [rbp-348h] BYREF
  unsigned __int64 v114; // [rsp+110h] [rbp-300h] BYREF
  __int64 v115; // [rsp+118h] [rbp-2F8h]
  __int64 v116; // [rsp+120h] [rbp-2F0h]
  unsigned int v117; // [rsp+128h] [rbp-2E8h] BYREF
  char v118; // [rsp+130h] [rbp-2E0h]
  char v119; // [rsp+131h] [rbp-2DFh]
  unsigned __int64 v120; // [rsp+170h] [rbp-2A0h] BYREF
  __int64 v121; // [rsp+178h] [rbp-298h]
  unsigned __int64 v122; // [rsp+180h] [rbp-290h] BYREF
  char v123[8]; // [rsp+188h] [rbp-288h] BYREF
  char v124; // [rsp+190h] [rbp-280h]
  char v125; // [rsp+191h] [rbp-27Fh]
  __int64 v126; // [rsp+1D0h] [rbp-240h] BYREF
  __int64 v127; // [rsp+1D8h] [rbp-238h]
  _BYTE v128[560]; // [rsp+1E0h] [rbp-230h] BYREF

  v2 = a1;
  sub_A4DCE0(&v126, a2 + 4, 10, 0);
  if ( (v126 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v126 & 0xFFFFFFFFFFFFFFFELL | 1;
    return v2;
  }
  if ( a2[193] )
  {
    v128[17] = 1;
    v126 = (__int64)"Invalid multiple blocks";
    v128[16] = 3;
    sub_9C81F0(a1, (__int64)(a2 + 1), (__int64)&v126);
    return v2;
  }
  v4 = &v106;
  v5 = (__int64)(a2 + 4);
  v126 = (__int64)v128;
  v127 = 0x4000000000LL;
LABEL_4:
  v7 = (unsigned __int64 *)v5;
  sub_9CEFB0((__int64)v4, v5, 0, v3);
  v8 = v107 & 1;
  v107 = (2 * (v107 & 1)) | v107 & 0xFD;
  if ( v8 )
  {
    v7 = (unsigned __int64 *)v4;
    v11 = v4;
    sub_9C9090(v2, v4);
    goto LABEL_17;
  }
  if ( (_DWORD)v106 == 1 )
  {
    *v2 = 1;
    goto LABEL_21;
  }
  if ( (v106 & 0xFFFFFFFD) == 0 )
  {
    v11 = v4;
    v7 = a2 + 1;
    v125 = 1;
    v120 = (unsigned __int64)"Malformed block";
    v124 = 3;
    sub_9C81F0(v2, (__int64)(a2 + 1), (__int64)&v120);
    goto LABEL_17;
  }
  LODWORD(v127) = 0;
  sub_A4B600(&v108, v5, HIDWORD(v106), &v126, 0);
  v9 = v109 & 1;
  v3 = (unsigned int)(2 * v9);
  v109 = (2 * v9) | v109 & 0xFD;
  if ( (_BYTE)v9 )
  {
    v7 = (unsigned __int64 *)&v108;
    v11 = v4;
    sub_9C8CD0(v2, &v108);
    goto LABEL_25;
  }
  if ( (_DWORD)v108 != 3 )
    goto LABEL_9;
  if ( (unsigned int)v127 <= 2 )
  {
    v11 = v4;
    v7 = a2 + 1;
    v125 = 1;
    v120 = (unsigned __int64)"Invalid grp record";
    v124 = 3;
    sub_9C81F0(v2, (__int64)(a2 + 1), (__int64)&v120);
    goto LABEL_25;
  }
  v12 = v126;
  v93 = 255;
  v13 = *(_QWORD *)v126;
  v91 = v5;
  v103 = 2;
  v92 = v13;
  v97 = v2;
  v99 = *(_QWORD *)(v126 + 8);
  v96 = v4;
  v14 = v127;
  v110 = a2[54];
  v111 = v113;
  v112 = 0x800000000LL;
  v15 = 2;
  while ( 1 )
  {
    v16 = (__int64 *)(v12 + 8LL * v15);
    v17 = *v16;
    if ( !*v16 )
    {
      v103 = v15 + 1;
      v18 = v15 + 1;
      v19 = *(_QWORD *)(v12 + 8 * v18);
      if ( v99 == 0xFFFFFFFFLL )
      {
        switch ( v19 )
        {
          case 20LL:
            v93 = 0;
            goto LABEL_75;
          case 21LL:
            v93 &= 0x55u;
            goto LABEL_75;
          case 45LL:
            v93 &= 3u;
            goto LABEL_75;
          case 49LL:
            v93 &= 0xCu;
            goto LABEL_75;
          case 50LL:
            v93 &= 0xFu;
            goto LABEL_75;
          case 52LL:
            v93 &= 0xAAu;
            goto LABEL_75;
          default:
            break;
        }
      }
      if ( v19 != 11 )
      {
        v7 = a2;
        sub_9C8460((__int64 *)&v120, (__int64)a2, v19, &v114);
        v20 = v120 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v120 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        {
          v2 = v97;
          v120 = 0;
          v11 = v96;
          *v97 = v20 | 1;
          sub_9C66B0((__int64 *)&v120);
          goto LABEL_36;
        }
        switch ( (_DWORD)v114 )
        {
          case 'Q':
            sub_A77E90(&v110, 0);
            break;
          case 'U':
            sub_A77EA0(&v110, 0);
            break;
          case 'S':
            sub_A77EB0(&v110, 0);
            break;
          case '_':
            sub_A77CB0(&v110, 2);
            break;
          default:
            if ( (unsigned int)(v114 - 1) > 0x4E )
            {
              v125 = 1;
              v2 = v97;
              v79 = a2;
              v80 = "Not an enum attribute";
              v11 = v96;
              goto LABEL_220;
            }
            sub_A77B20(&v110, (unsigned int)v114);
            break;
        }
        goto LABEL_74;
      }
      sub_A77CE0(&v110, 0);
      LODWORD(v18) = v103;
      goto LABEL_75;
    }
    if ( v17 == 1 )
    {
      v50 = v15 + 1;
      v103 = v50;
      sub_9C8460((__int64 *)&v120, (__int64)a2, *(_QWORD *)(v12 + 8 * v50), &v114);
      v21 = v120 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v120 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_82;
      if ( (unsigned int)(v114 - 86) > 0xA )
      {
        v125 = 1;
        v2 = v97;
        v79 = a2;
        v80 = "Not an int attribute";
        v11 = v96;
        goto LABEL_220;
      }
      LODWORD(v18) = v103;
      v51 = v103 + 1;
      switch ( (_DWORD)v114 )
      {
        case 'V':
          ++v103;
          v86 = (unsigned __int8)v86;
          if ( *(_DWORD *)(v126 + 8 * v51) )
          {
            _BitScanReverse64(&v73, *(unsigned int *)(v126 + 8 * v51));
            LOBYTE(v74) = 63 - (v73 ^ 0x3F);
            HIBYTE(v74) = 1;
            v86 = v74;
          }
          sub_A77B90(&v110, v86);
          LODWORD(v18) = v103;
          break;
        case '^':
          ++v103;
          v87 = (unsigned __int8)v87;
          if ( *(_DWORD *)(v126 + 8 * v51) )
          {
            _BitScanReverse64(&v75, *(unsigned int *)(v126 + 8 * v51));
            LOBYTE(v76) = 63 - (v75 ^ 0x3F);
            HIBYTE(v76) = 1;
            v87 = v76;
          }
          sub_A77BC0(&v110, v87);
          LODWORD(v18) = v103;
          break;
        case 'Z':
          ++v103;
          sub_A77BF0(&v110, *(_QWORD *)(v126 + 8 * v51));
          LODWORD(v18) = v103;
          break;
        case '[':
          ++v103;
          sub_A77C10(&v110, *(_QWORD *)(v126 + 8 * v51));
          LODWORD(v18) = v103;
          break;
        case 'X':
          ++v103;
          sub_A77C30(&v110, *(_QWORD *)(v126 + 8 * v51));
          LODWORD(v18) = v103;
          break;
        case '`':
          ++v103;
          sub_A77C60(&v110, *(_QWORD *)(v126 + 8 * v51));
          LODWORD(v18) = v103;
          break;
        case '_':
          ++v103;
          sub_A77CB0(&v110, *(_QWORD *)(v126 + 8 * v51));
          LODWORD(v18) = v103;
          break;
        case 'W':
          ++v103;
          sub_A77D20(&v110, *(_QWORD *)(v126 + 8 * v51));
          LODWORD(v18) = v103;
          break;
        case '\\':
          ++v103;
          v77 = *(_QWORD *)(v126 + 8 * v51);
          if ( HIBYTE(v77) )
            sub_A77CD0(&v110, v77);
          else
            sub_A77CD0(
              &v110,
              (4 * ((v77 >> 2) & 3)) | v77 & 3 | (((v77 >> 4) & 3) << 6) | (16 * ((unsigned int)(v77 >> 4) & 3)));
          LODWORD(v18) = v103;
          break;
        case 'Y':
          ++v103;
          v81 = *(_QWORD *)(v126 + 8 * v51);
          v82 = v85;
          LOBYTE(v82) = (unsigned int)v81 >> 4;
          BYTE1(v82) = v81 & 0xF;
          v85 = v82;
          sub_A77CE0(&v110, v82);
          LODWORD(v18) = v103;
          break;
        case ']':
          ++v103;
          sub_A77D00(&v110, *(_QWORD *)(v126 + 8 * v51) & 0x3FFLL);
          goto LABEL_74;
      }
      goto LABEL_75;
    }
    if ( (unsigned __int64)(v17 - 3) <= 1 )
    {
      v38 = v15 + 1;
      v103 = v38;
      v39 = *v16;
      v40 = (__int64 *)(v12 + 8LL * v38);
      v114 = (unsigned __int64)&v117;
      v115 = 0;
      v116 = 64;
      v120 = (unsigned __int64)v123;
      v121 = 0;
      v122 = 64;
      if ( *v40 )
      {
        v41 = 0;
        v42 = 64;
        v43 = &v114;
        if ( v38 != v14 )
        {
          while ( 1 )
          {
            v103 = v38 + 1;
            v44 = *v40;
            if ( v42 < v41 + 1 )
            {
              v88 = *v40;
              v95 = v43;
              sub_C8D290(v43, &v117, v41 + 1, 1);
              v41 = v115;
              LOBYTE(v44) = v88;
              v43 = v95;
            }
            *(_BYTE *)(v114 + v41) = v44;
            v12 = v126;
            v41 = v115 + 1;
            v38 = v103;
            v40 = (__int64 *)(v126 + 8LL * v103);
            ++v115;
            if ( !*v40 || v103 == v14 )
              break;
            v42 = v116;
          }
        }
      }
      v45 = v121;
      if ( v39 == 4 )
      {
        v45 = v121;
        v103 = v38 + 1;
        v60 = v38 + 1;
        for ( i = (__int64 *)(v12 + 8 * v60); *i; LODWORD(v60) = v103 )
        {
          if ( (_DWORD)v60 == v14 )
            break;
          v103 = v60 + 1;
          v62 = *i;
          if ( v45 + 1 > v122 )
          {
            sub_C8D290(&v120, v123, v45 + 1, 1);
            v45 = v121;
          }
          *(_BYTE *)(v120 + v45) = v62;
          v45 = v121 + 1;
          i = (__int64 *)(v126 + 8LL * v103);
          ++v121;
        }
      }
      v46 = v114;
      sub_A78980(&v110, v114, v115, v120, v45);
      if ( (char *)v120 != v123 )
        _libc_free(v120, v46);
      v47 = v114;
      if ( (unsigned int *)v114 == &v117 )
        goto LABEL_74;
LABEL_95:
      _libc_free(v47, v46);
LABEL_74:
      LODWORD(v18) = v103;
      goto LABEL_75;
    }
    if ( (unsigned __int64)(v17 - 5) > 1 )
      break;
    v37 = v15 + 1;
    v103 = v37;
    sub_9C8460((__int64 *)&v120, (__int64)a2, *(_QWORD *)(v12 + 8 * v37), &v114);
    if ( (v120 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v2 = v97;
      v11 = v96;
      v120 = v120 & 0xFFFFFFFFFFFFFFFELL | 1;
      *v97 = 0;
      goto LABEL_81;
    }
    v48 = (unsigned int)v114;
    if ( (unsigned int)(v114 - 80) > 5 )
    {
      v125 = 1;
      v2 = v97;
      v79 = a2;
      v80 = "Not a type attribute";
      v11 = v96;
      goto LABEL_220;
    }
    v49 = 0;
    if ( v17 == 6 )
    {
      v59 = sub_9CAD80(a2, *(_QWORD *)(v126 + 8LL * ++v103));
      v48 = (unsigned int)v114;
      v49 = v59;
    }
    sub_A77E60(&v110, v48, v49);
    LODWORD(v18) = v103;
LABEL_75:
    v15 = v18 + 1;
    v103 = v15;
    if ( v15 == v14 )
    {
      v5 = v91;
      v2 = v97;
      v4 = v96;
      if ( v93 != 255 )
        sub_A77CD0(&v110, v93);
      sub_A88D70(&v110);
      v54 = v92;
      v55 = sub_A7B020(a2[54], (unsigned int)v99, &v110);
      v56 = a2[190];
      v57 = a2 + 189;
      if ( !v56 )
        goto LABEL_165;
      do
      {
        v3 = *(_QWORD *)(v56 + 16);
        v58 = *(_QWORD *)(v56 + 24);
        if ( v92 > *(_DWORD *)(v56 + 32) )
        {
          v56 = *(_QWORD *)(v56 + 24);
        }
        else
        {
          v57 = (unsigned __int64 *)v56;
          v56 = *(_QWORD *)(v56 + 16);
        }
      }
      while ( v56 );
      if ( v57 == a2 + 189 || v92 < *((_DWORD *)v57 + 8) )
      {
LABEL_165:
        v98 = a2 + 189;
        v100 = (__int64)v57;
        v63 = sub_22077B0(48);
        *(_QWORD *)(v63 + 40) = 0;
        *(_DWORD *)(v63 + 32) = v92;
        v101 = v63;
        v64 = sub_9D5900(a2 + 188, v100, (unsigned int *)(v63 + 32));
        if ( v65 )
        {
          v66 = v98 == (_QWORD *)v65 || v64 != 0;
          if ( !v66 )
            v66 = v92 < *(_DWORD *)(v65 + 32);
          v54 = v101;
          sub_220F040(v66, v101, v65, v98);
          v57 = (unsigned __int64 *)v101;
          ++a2[193];
        }
        else
        {
          v78 = v101;
          v54 = 48;
          v102 = v64;
          j_j___libc_free_0(v78, 48);
          v57 = (unsigned __int64 *)v102;
        }
      }
      v57[5] = v55;
      if ( v111 != v113 )
        _libc_free(v111, v54);
      if ( (v109 & 2) != 0 )
        goto LABEL_122;
      if ( (v109 & 1) != 0 && v108 )
        (*(void (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v108 + 8LL))(v108, v54, v58, v3);
LABEL_9:
      if ( (v107 & 2) != 0 )
      {
        v11 = v4;
        goto LABEL_78;
      }
      if ( (v107 & 1) != 0 )
      {
        if ( v106 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v106 + 8LL))(v106);
      }
      goto LABEL_4;
    }
    v12 = v126;
  }
  if ( v17 == 7 )
  {
    v103 = v15 + 2;
    sub_9C8460((__int64 *)&v120, (__int64)a2, *(_QWORD *)(v12 + 8LL * (v15 + 1)), &v104);
    v21 = v120 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v120 & 0xFFFFFFFFFFFFFFFELL) == 0 )
    {
      if ( v104 != 97 )
      {
        v125 = 1;
        v2 = v97;
        v79 = a2;
        v80 = "Not a ConstantRange attribute";
        v11 = v96;
LABEL_220:
        v7 = v79 + 1;
        v120 = (unsigned __int64)v80;
        v124 = 3;
        sub_9C81F0(v2, (__int64)(v79 + 1), (__int64)&v120);
        goto LABEL_36;
      }
      v52 = v103;
      if ( (unsigned int)v127 == (unsigned __int64)v103 )
      {
        v125 = 1;
        v120 = (unsigned __int64)"Too few records for range";
        v2 = v97;
        v124 = 3;
        v11 = v96;
        sub_9C81F0(&v105, (__int64)(a2 + 1), (__int64)&v120);
        v118 |= 3u;
        v114 = v105 & 0xFFFFFFFFFFFFFFFELL;
LABEL_200:
        v7 = &v114;
        sub_9C94E0(v2, (__int64 *)&v114);
        sub_9D2150(&v114);
        goto LABEL_36;
      }
      ++v103;
      sub_9C8500((__int64)&v114, (__int64)a2, v126, (unsigned int)v127, &v103, *(_DWORD *)(v126 + 8 * v52));
      v53 = v118 & 1;
      v118 = (2 * (v118 & 1)) | v118 & 0xFD;
      if ( v53 )
      {
        v2 = v97;
        v11 = v96;
        goto LABEL_200;
      }
      --v103;
      sub_A78BB0(&v110, v104, &v114);
      if ( (v118 & 2) != 0 )
LABEL_141:
        sub_9D20E0(&v114);
      if ( (v118 & 1) != 0 )
      {
        if ( v114 )
          (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v114 + 8LL))(v114);
      }
      else
      {
        if ( v117 > 0x40 && v116 )
          j_j___libc_free_0_0(v116);
        if ( (unsigned int)v115 > 0x40 && v114 )
          j_j___libc_free_0_0(v114);
      }
      goto LABEL_74;
    }
LABEL_82:
    v2 = v97;
    v11 = v96;
    v120 = v21 | 1;
    *v97 = 0;
LABEL_81:
    v7 = &v120;
    sub_9C6670(v2, &v120);
    sub_9C66B0((__int64 *)&v120);
    goto LABEL_36;
  }
  if ( v17 != 8 )
  {
    v2 = v97;
    v125 = 1;
    v7 = a2 + 1;
    v120 = (unsigned __int64)"Invalid attribute group entry";
    v11 = v96;
    v124 = 3;
    sub_9C81F0(v97, (__int64)(a2 + 1), (__int64)&v120);
    goto LABEL_36;
  }
  v103 = v15 + 2;
  sub_9C8460((__int64 *)&v120, (__int64)a2, *(_QWORD *)(v12 + 8LL * (v15 + 1)), &v105);
  v21 = v120 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v120 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    goto LABEL_82;
  if ( (_DWORD)v105 != 98 )
  {
    v125 = 1;
    v2 = v97;
    v79 = a2;
    v80 = "Not a constant range list attribute";
    v11 = v96;
    goto LABEL_220;
  }
  v120 = (unsigned __int64)&v122;
  v121 = 0x200000000LL;
  v22 = v103;
  v23 = v103 + 2;
  if ( v103 + 2 > v14 )
  {
    v119 = 1;
    v2 = v97;
    v83 = a2;
    v84 = "Too few records for constant range list";
    v11 = v96;
    goto LABEL_226;
  }
  v24 = v126;
  v25 = ++v103;
  v26 = *(_QWORD *)(v126 + 8 * v22);
  v103 = v23;
  v27 = *(_DWORD *)(v126 + 8 * v25);
  v94 = v26;
  if ( (_DWORD)v26 )
  {
    v28 = 0;
    while ( 1 )
    {
      v7 = a2;
      sub_9C8500((__int64)&v114, (__int64)a2, v24, (unsigned int)v127, &v103, v27);
      v29 = v118 & 1;
      v118 = (2 * (v118 & 1)) | v118 & 0xFD;
      if ( v29 )
        break;
      v30 = (unsigned int)v121;
      v31 = &v114;
      v32 = v120;
      v33 = v121;
      if ( (unsigned __int64)(unsigned int)v121 + 1 > HIDWORD(v121) )
      {
        if ( v120 > (unsigned __int64)&v114 || (unsigned __int64)&v114 >= v120 + 32LL * (unsigned int)v121 )
        {
          sub_9D5330((__int64)&v120, (unsigned int)v121 + 1LL);
          v30 = (unsigned int)v121;
          v32 = v120;
          v31 = &v114;
          v33 = v121;
        }
        else
        {
          v90 = (char *)&v114 - v120;
          sub_9D5330((__int64)&v120, (unsigned int)v121 + 1LL);
          v32 = v120;
          v30 = (unsigned int)v121;
          v31 = (unsigned __int64 *)&v90[v120];
          v33 = v121;
        }
      }
      v34 = 32 * v30 + v32;
      if ( v34 )
      {
        v35 = *((_DWORD *)v31 + 2);
        *(_DWORD *)(v34 + 8) = v35;
        if ( v35 > 0x40 )
        {
          v89 = v34;
          sub_C43780(v34, v31);
          v34 = v89;
        }
        else
        {
          *(_QWORD *)v34 = *v31;
        }
        v36 = *((_DWORD *)v31 + 6);
        *(_DWORD *)(v34 + 24) = v36;
        if ( v36 > 0x40 )
        {
          v31 += 2;
          sub_C43780(v34 + 16, v31);
        }
        else
        {
          *(_QWORD *)(v34 + 16) = v31[2];
        }
        v33 = v121;
      }
      LODWORD(v121) = v33 + 1;
      if ( (v118 & 2) != 0 )
        goto LABEL_141;
      if ( (v118 & 1) != 0 )
      {
        if ( v114 )
          (*(void (__fastcall **)(unsigned __int64, unsigned __int64 *))(*(_QWORD *)v114 + 8LL))(v114, v31);
      }
      else
      {
        if ( v117 > 0x40 && v116 )
          j_j___libc_free_0_0(v116);
        if ( (unsigned int)v115 > 0x40 && v114 )
          j_j___libc_free_0_0(v114);
      }
      if ( v94 == ++v28 )
        goto LABEL_175;
      v24 = v126;
    }
    v2 = v97;
    v11 = v96;
    *v97 = v114 | 1;
    goto LABEL_188;
  }
LABEL_175:
  --v103;
  if ( (unsigned __int8)sub_ABEE90(v120, (unsigned int)v121) )
  {
    v46 = (unsigned int)v105;
    sub_A79080(&v110, (unsigned int)v105, v120, (unsigned int)v121);
    v67 = (unsigned __int64 *)v120;
    v68 = (unsigned __int64 *)(v120 + 32LL * (unsigned int)v121);
    if ( (unsigned __int64 *)v120 != v68 )
    {
      do
      {
        v68 -= 4;
        if ( *((_DWORD *)v68 + 6) > 0x40u )
        {
          v69 = v68[2];
          if ( v69 )
            j_j___libc_free_0_0(v69);
        }
        if ( *((_DWORD *)v68 + 2) > 0x40u && *v68 )
          j_j___libc_free_0_0(*v68);
      }
      while ( v67 != v68 );
      v67 = (unsigned __int64 *)v120;
    }
    v47 = (unsigned __int64)v67;
    if ( v67 == &v122 )
      goto LABEL_74;
    goto LABEL_95;
  }
  v119 = 1;
  v2 = v97;
  v83 = a2;
  v84 = "Invalid (unordered or overlapping) range list";
  v11 = v96;
LABEL_226:
  v7 = v83 + 1;
  v114 = (unsigned __int64)v84;
  v118 = 3;
  sub_9C81F0(v2, (__int64)(v83 + 1), (__int64)&v114);
LABEL_188:
  v70 = (unsigned __int64 *)v120;
  v71 = (unsigned __int64 *)(v120 + 32LL * (unsigned int)v121);
  if ( (unsigned __int64 *)v120 != v71 )
  {
    do
    {
      v71 -= 4;
      if ( *((_DWORD *)v71 + 6) > 0x40u )
      {
        v72 = v71[2];
        if ( v72 )
          j_j___libc_free_0_0(v72);
      }
      if ( *((_DWORD *)v71 + 2) > 0x40u && *v71 )
        j_j___libc_free_0_0(*v71);
    }
    while ( v70 != v71 );
    v70 = (unsigned __int64 *)v120;
  }
  if ( v70 != &v122 )
    _libc_free(v70, v7);
LABEL_36:
  if ( v111 != v113 )
    _libc_free(v111, v7);
LABEL_25:
  if ( (v109 & 2) != 0 )
LABEL_122:
    sub_9CE230(&v108);
  if ( (v109 & 1) != 0 && v108 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v108 + 8LL))(v108);
LABEL_17:
  if ( (v107 & 2) != 0 )
LABEL_78:
    sub_9CEF10(v11);
  if ( (v107 & 1) != 0 && v106 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v106 + 8LL))(v106);
LABEL_21:
  if ( (_BYTE *)v126 != v128 )
    _libc_free(v126, v7);
  return v2;
}
