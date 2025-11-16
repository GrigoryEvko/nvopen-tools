// Function: sub_F54050
// Address: 0xf54050
//
__int64 *__fastcall sub_F54050(unsigned __int8 *a1, __int64 *a2, __int64 a3, __int64 *a4, __int64 a5)
{
  __int64 v6; // rax
  _BYTE *v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rdx
  __int64 v11; // rbx
  _QWORD *v12; // r15
  __int64 *v13; // rax
  unsigned __int8 *v14; // r14
  __int64 v15; // rax
  __int64 v16; // rcx
  unsigned __int64 v17; // rdx
  int v18; // r10d
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rax
  __int64 *v21; // rax
  __int64 v22; // r13
  __int64 v23; // rax
  __int64 v24; // rdx
  _QWORD *v25; // rsi
  int v26; // edx
  __int64 *result; // rax
  char v28; // al
  _BYTE *v29; // rsi
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // rdx
  __int64 v33; // rbx
  __int64 *v34; // r15
  unsigned __int8 *v35; // rax
  unsigned __int8 *v36; // r14
  _QWORD *v37; // r12
  unsigned __int8 *v38; // r15
  __int64 v39; // rax
  __int64 v40; // rcx
  unsigned __int64 v41; // rdx
  int v42; // r10d
  unsigned __int64 v43; // rax
  unsigned __int64 v44; // rax
  __int64 *v45; // rax
  unsigned __int8 *v46; // rax
  _QWORD *v47; // r15
  __int64 v48; // rax
  unsigned __int8 *v49; // rsi
  __int64 v50; // r15
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rax
  unsigned int v54; // eax
  __int64 *v55; // rbx
  __int64 v56; // rdi
  __int64 *v57; // rbx
  __int64 v58; // rdi
  unsigned __int8 *v59; // rdi
  unsigned __int8 *v60; // rsi
  __int64 v61; // rax
  __int64 v62; // r15
  unsigned __int8 *v63; // rbx
  __int64 v64; // r14
  _QWORD *v65; // rax
  _QWORD *v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // r8
  __int64 v70; // r15
  __int64 v71; // rsi
  __int64 *v72; // rdi
  __int64 *v73; // rdi
  __int64 v74; // rdx
  __int64 v75; // r15
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 v78; // rsi
  unsigned __int8 *v79; // rdi
  __int64 v80; // rsi
  __int64 v81; // r15
  __int64 *v82; // rsi
  _QWORD *v83; // rax
  __int64 v84; // rdx
  __int64 v85; // rcx
  __int64 v86; // r8
  __int64 v87; // rax
  __int64 v88; // rdx
  __int64 v89; // rcx
  __int64 v90; // r14
  __int64 *v91; // rdi
  __int64 v92; // rdx
  __int64 *v93; // rax
  __int64 v94; // rcx
  __int64 v95; // rcx
  __int64 *v96; // rdi
  __int64 v99; // [rsp+18h] [rbp-168h]
  __int64 *v100; // [rsp+28h] [rbp-158h]
  __int64 *v102; // [rsp+30h] [rbp-150h]
  __int64 *v103; // [rsp+40h] [rbp-140h]
  __int64 *v104; // [rsp+40h] [rbp-140h]
  __int64 v105; // [rsp+48h] [rbp-138h]
  __int64 v106; // [rsp+48h] [rbp-138h]
  char v107; // [rsp+53h] [rbp-12Dh]
  bool v108; // [rsp+54h] [rbp-12Ch]
  bool v109; // [rsp+54h] [rbp-12Ch]
  int v110; // [rsp+58h] [rbp-128h]
  int v111; // [rsp+58h] [rbp-128h]
  __int64 v112; // [rsp+58h] [rbp-128h]
  _QWORD *v113; // [rsp+58h] [rbp-128h]
  __int64 *v114; // [rsp+80h] [rbp-100h] BYREF
  __int64 *v115; // [rsp+88h] [rbp-F8h]
  _QWORD *v116; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v117; // [rsp+98h] [rbp-E8h]
  _BYTE v118[32]; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 *i; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v120; // [rsp+C8h] [rbp-B8h]
  _BYTE v121[176]; // [rsp+D0h] [rbp-B0h] BYREF

  v100 = &a2[a3];
  if ( v100 == a2 )
  {
    result = a4;
    v102 = &a4[a5];
    if ( a4 != v102 )
    {
      v107 = 0;
      goto LABEL_50;
    }
    return result;
  }
  v103 = a2;
  v107 = 0;
  while ( 1 )
  {
    v6 = *(_QWORD *)(*v103 - 32);
    v105 = *v103;
    if ( !v6 )
      BUG();
    if ( *(_BYTE *)v6 || *(_QWORD *)(v6 + 24) != *(_QWORD *)(*v103 + 80) )
LABEL_185:
      BUG();
    if ( *(_DWORD *)(v6 + 36) != 68 )
      break;
    if ( a1 != (unsigned __int8 *)sub_B595C0(v105) )
      goto LABEL_100;
    v79 = (unsigned __int8 *)sub_B595C0(v105);
    if ( *v79 > 0x1Cu )
    {
      v80 = 0;
      v116 = v118;
      v117 = 0x400000000LL;
      i = (__int64 *)v121;
      v120 = 0x1000000000LL;
      v81 = sub_F53E50(v79, 0, (__int64)&i, (__int64)&v116);
      if ( !v81 )
        goto LABEL_172;
      v82 = i;
      v83 = (_QWORD *)sub_B0DBA0(
                        *(_QWORD **)(*(_QWORD *)(v105 + 32 * (5LL - (*(_DWORD *)(v105 + 4) & 0x7FFFFFF))) + 24LL),
                        i,
                        (unsigned int)v120,
                        0,
                        0);
      v87 = sub_E3D320(v83, (__int64)v82, v84, v85, v86);
      v80 = (unsigned int)v117;
      v90 = v87;
      if ( (_DWORD)v117 )
      {
        sub_B59B20(v105);
        v96 = i;
        if ( i != (__int64 *)v121 )
          goto LABEL_173;
      }
      else
      {
        sub_B59690(v105, v81, v88, v89);
        v91 = (__int64 *)(*(_QWORD *)(v90 + 8) & 0xFFFFFFFFFFFFFFF8LL);
        if ( (*(_QWORD *)(v90 + 8) & 4) != 0 )
          v91 = (__int64 *)*v91;
        v80 = v105;
        v92 = sub_B9F6F0(v91, (_BYTE *)v90);
        v93 = (__int64 *)(v105 + 32 * (5LL - (*(_DWORD *)(v105 + 4) & 0x7FFFFFF)));
        if ( *v93 )
        {
          v80 = v93[2];
          v94 = v93[1];
          *(_QWORD *)v80 = v94;
          if ( v94 )
          {
            v80 = v93[2];
            *(_QWORD *)(v94 + 16) = v80;
          }
        }
        *v93 = v92;
        if ( v92 )
        {
          v95 = *(_QWORD *)(v92 + 16);
          v80 = v92 + 16;
          v93[1] = v95;
          if ( v95 )
            *(_QWORD *)(v95 + 16) = v93 + 1;
          v93[2] = v80;
          *(_QWORD *)(v92 + 16) = v93;
        }
LABEL_172:
        v96 = i;
        if ( i != (__int64 *)v121 )
LABEL_173:
          _libc_free(v96, v80);
      }
      if ( v116 != (_QWORD *)v118 )
        _libc_free(v116, v80);
    }
    v107 = 1;
LABEL_100:
    if ( a1 == (unsigned __int8 *)sub_B58EB0(v105, 0) )
    {
      v6 = *(_QWORD *)(v105 - 32);
      if ( !v6 || *(_BYTE *)v6 )
        goto LABEL_185;
      break;
    }
LABEL_48:
    if ( v100 == ++v103 )
      goto LABEL_49;
  }
  v7 = (_BYTE *)v105;
  if ( *(_QWORD *)(v6 + 24) != *(_QWORD *)(v105 + 80) )
    goto LABEL_185;
  v108 = *(_DWORD *)(v6 + 36) == 71 || *(_DWORD *)(v6 + 36) == 68;
  sub_B58E30(&v114, v105);
  v116 = v118;
  v117 = 0x400000000LL;
  v10 = (__int64)v115;
  v11 = (__int64)v114;
  v12 = *(_QWORD **)(*(_QWORD *)(v105 + 32 * (2LL - (*(_DWORD *)(v105 + 4) & 0x7FFFFFF))) + 24LL);
  for ( i = v114; v115 != (__int64 *)v11; v11 = (unsigned __int64)(v13 + 1) | 4 )
  {
    while ( 1 )
    {
      v13 = (__int64 *)(v11 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v11 & 4) != 0 )
        break;
      if ( a1 != (unsigned __int8 *)v13[17] )
      {
        v11 = (__int64)(v13 + 18);
        if ( v115 != v13 + 18 )
          continue;
      }
      goto LABEL_15;
    }
    v8 = *v13;
    if ( a1 == *(unsigned __int8 **)(*v13 + 136) )
      break;
  }
LABEL_15:
  if ( !v12 )
    goto LABEL_49;
  v14 = 0;
  while ( 1 )
  {
    if ( (__int64 *)v11 == v115 )
    {
      if ( !v14 )
        goto LABEL_117;
LABEL_37:
      v22 = sub_E3D320(v12, (__int64)v7, v10, v8, v9);
      sub_B59720(v105, (__int64)a1, v14);
      v23 = (__int64)(*(_QWORD *)(v22 + 24) - *(_QWORD *)(v22 + 16)) >> 3;
      if ( (_DWORD)v117 || (unsigned int)v23 > 0x80 )
      {
        v24 = *(_QWORD *)(v105 - 32);
        if ( !v24 )
          goto LABEL_185;
        if ( *(_BYTE *)v24 )
          goto LABEL_185;
        v25 = *(_QWORD **)(v105 + 80);
        if ( *(_QWORD **)(v24 + 24) != v25 )
          goto LABEL_185;
        v26 = *(_DWORD *)(v24 + 36);
        if ( v26 != 68 && v26 != 71 || (unsigned int)v23 > 0x80 )
          goto LABEL_44;
        v25 = (_QWORD *)v105;
        v52 = *(_QWORD *)(*(_QWORD *)(v105 - 32LL * (*(_DWORD *)(v105 + 4) & 0x7FFFFFF)) + 24LL);
        v53 = 1;
        if ( *(_BYTE *)v52 == 4 )
          v53 = *(unsigned int *)(v52 + 144);
        if ( (unsigned __int64)(unsigned int)v117 + v53 <= 0x10 )
        {
          v25 = v116;
          sub_B59230(v105, v116, (_BYTE *)(unsigned int)v117, v22);
        }
        else
        {
LABEL_44:
          sub_F507F0(v105);
        }
      }
      else
      {
        v73 = (__int64 *)(*(_QWORD *)(v22 + 8) & 0xFFFFFFFFFFFFFFF8LL);
        if ( (*(_QWORD *)(v22 + 8) & 4) != 0 )
          v73 = (__int64 *)*v73;
        v25 = (_QWORD *)v105;
        v74 = sub_B9F6F0(v73, (_BYTE *)v22);
        v75 = v105 + 32 * (2LL - (*(_DWORD *)(v105 + 4) & 0x7FFFFFF));
        if ( *(_QWORD *)v75 )
        {
          v76 = *(_QWORD *)(v75 + 8);
          **(_QWORD **)(v75 + 16) = v76;
          if ( v76 )
            *(_QWORD *)(v76 + 16) = *(_QWORD *)(v75 + 16);
        }
        *(_QWORD *)v75 = v74;
        if ( v74 )
        {
          v77 = *(_QWORD *)(v74 + 16);
          *(_QWORD *)(v75 + 8) = v77;
          if ( v77 )
          {
            v25 = (_QWORD *)(v75 + 8);
            *(_QWORD *)(v77 + 16) = v75 + 8;
          }
          *(_QWORD *)(v75 + 16) = v74 + 16;
          *(_QWORD *)(v74 + 16) = v75;
        }
      }
      if ( v116 != (_QWORD *)v118 )
        _libc_free(v116, v25);
      v107 = 1;
      goto LABEL_48;
    }
    v120 = 0x1000000000LL;
    v15 = (__int64)v114;
    i = (__int64 *)v121;
    if ( (__int64 *)v11 == v114 )
    {
      v18 = 0;
    }
    else
    {
      v16 = 0;
      do
      {
        while ( 1 )
        {
          v17 = v15 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v15 & 4) != 0 )
            break;
          ++v16;
          v15 = v17 + 144;
          if ( v11 == v17 + 144 )
            goto LABEL_23;
        }
        ++v16;
        v15 = (v17 + 8) | 4;
      }
      while ( v11 != v15 );
LABEL_23:
      v18 = v16;
    }
    v110 = v18;
    v7 = (_BYTE *)sub_AF4EB0((__int64)v12);
    v14 = (unsigned __int8 *)sub_F53E50(a1, (__int64)v7, (__int64)&i, (__int64)&v116);
    if ( !v14 )
      break;
    v7 = i;
    v12 = (_QWORD *)sub_B0DBA0(v12, i, (unsigned int)v120, v110, v108);
    v19 = v11 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v11 & 4) != 0 )
    {
      v11 = (v19 + 8) | 4;
      v20 = v11;
    }
    else
    {
      v20 = v19 + 144;
      v11 = v20;
    }
    v10 = (__int64)v115;
    if ( v115 != (__int64 *)v20 )
    {
      do
      {
        while ( 1 )
        {
          v21 = (__int64 *)(v11 & 0xFFFFFFFFFFFFFFF8LL);
          if ( (v11 & 4) != 0 )
            break;
          if ( a1 != (unsigned __int8 *)v21[17] )
          {
            v11 = (__int64)(v21 + 18);
            if ( v115 != v21 + 18 )
              continue;
          }
          goto LABEL_34;
        }
        v8 = *v21;
        if ( a1 == *(unsigned __int8 **)(*v21 + 136) )
          break;
        v11 = (unsigned __int64)(v21 + 1) | 4;
      }
      while ( v115 != (__int64 *)v11 );
    }
LABEL_34:
    if ( i != (__int64 *)v121 )
      _libc_free(i, v7);
    if ( !v12 )
      goto LABEL_37;
  }
  if ( i != (__int64 *)v121 )
    _libc_free(i, v7);
LABEL_117:
  if ( v116 != (_QWORD *)v118 )
    _libc_free(v116, v7);
LABEL_49:
  result = a4;
  v102 = &a4[a5];
  if ( a4 == v102 )
  {
    if ( !v107 )
      goto LABEL_127;
    return result;
  }
LABEL_50:
  v104 = a4;
  while ( 2 )
  {
    v28 = *(_BYTE *)(*v104 + 64);
    v106 = *v104;
    if ( v28 == 2 )
    {
      if ( a1 != sub_B13320(*v104) )
      {
LABEL_52:
        if ( a1 == (unsigned __int8 *)sub_B12A50(v106, 0) )
        {
          v28 = *(_BYTE *)(v106 + 64);
          break;
        }
LABEL_53:
        result = ++v104;
        if ( v102 == v104 )
          goto LABEL_125;
        continue;
      }
      v59 = sub_B13320(v106);
      if ( *v59 <= 0x1Cu )
      {
LABEL_146:
        v107 = 1;
        goto LABEL_52;
      }
      v60 = 0;
      v116 = v118;
      v117 = 0x400000000LL;
      i = (__int64 *)v121;
      v120 = 0x1000000000LL;
      v61 = sub_F53E50(v59, 0, (__int64)&i, (__int64)&v116);
      if ( v61 )
      {
        v62 = (unsigned int)v120;
        v112 = v61;
        v63 = (unsigned __int8 *)i;
        v64 = v106 + 88;
        v65 = (_QWORD *)sub_B11F60(v106 + 88);
        v60 = v63;
        v66 = (_QWORD *)sub_B0DBA0(v65, v63, v62, 0, 0);
        v70 = sub_E3D320(v66, (__int64)v63, v67, v68, v69);
        if ( (_DWORD)v117 )
        {
          sub_B14010(v106, (__int64)v63);
          v72 = i;
          if ( i == (__int64 *)v121 )
          {
LABEL_144:
            if ( v116 != (_QWORD *)v118 )
              _libc_free(v116, v60);
            goto LABEL_146;
          }
LABEL_143:
          _libc_free(v72, v60);
          goto LABEL_144;
        }
        v113 = sub_B98A20(v112, v106);
        sub_B91340(v106 + 40, 1);
        *(_QWORD *)(v106 + 48) = v113;
        sub_B96F50(v106 + 40, 1);
        sub_B11F20(&v114, v70);
        v71 = *(_QWORD *)(v106 + 88);
        if ( v71 )
          sub_B91220(v64, v71);
        v60 = (unsigned __int8 *)v114;
        *(_QWORD *)(v106 + 88) = v114;
        if ( v60 )
          sub_B976B0((__int64)&v114, v60, v64);
      }
      v72 = i;
      if ( i == (__int64 *)v121 )
        goto LABEL_144;
      goto LABEL_143;
    }
    break;
  }
  v109 = v28 != 0;
  v29 = (_BYTE *)v106;
  sub_B129C0(&v114, v106);
  v116 = v118;
  v117 = 0x400000000LL;
  v99 = v106 + 80;
  result = (__int64 *)sub_B11F60(v106 + 80);
  v32 = (__int64)v115;
  v33 = (__int64)v114;
  v34 = result;
  i = v114;
  if ( v115 != v114 )
  {
    while ( 1 )
    {
      result = (__int64 *)(v33 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v33 & 4) != 0 )
        break;
      if ( a1 == (unsigned __int8 *)result[17] )
        goto LABEL_63;
      if ( result )
      {
        result += 18;
        v33 = (__int64)result;
        if ( v115 == result )
          goto LABEL_63;
      }
      else
      {
LABEL_58:
        v33 = (unsigned __int64)(result + 1) | 4;
        result = (__int64 *)v33;
        if ( v115 == (__int64 *)v33 )
          goto LABEL_63;
      }
    }
    v30 = *result;
    if ( a1 == *(unsigned __int8 **)(*result + 136) )
      goto LABEL_63;
    goto LABEL_58;
  }
LABEL_63:
  if ( v34 )
  {
    v35 = a1;
    v36 = 0;
    v37 = v34;
    v38 = v35;
    while ( (__int64 *)v33 != v115 )
    {
      v120 = 0x1000000000LL;
      v39 = (__int64)v114;
      i = (__int64 *)v121;
      if ( (__int64 *)v33 == v114 )
      {
        v42 = 0;
      }
      else
      {
        v40 = 0;
        do
        {
          while ( 1 )
          {
            v41 = v39 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (v39 & 4) != 0 || !v41 )
              break;
            ++v40;
            v39 = v41 + 144;
            if ( v41 + 144 == v33 )
              goto LABEL_72;
          }
          ++v40;
          v39 = (v41 + 8) | 4;
        }
        while ( v39 != v33 );
LABEL_72:
        v42 = v40;
      }
      v111 = v42;
      v29 = (_BYTE *)sub_AF4EB0((__int64)v37);
      result = (__int64 *)sub_F53E50(v38, (__int64)v29, (__int64)&i, (__int64)&v116);
      v36 = (unsigned __int8 *)result;
      if ( !result )
      {
        if ( i != (__int64 *)v121 )
          result = (__int64 *)_libc_free(i, v29);
        goto LABEL_123;
      }
      v29 = i;
      v37 = (_QWORD *)sub_B0DBA0(v37, i, (unsigned int)v120, v111, v109);
      v43 = v33 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v33 & 4) != 0 || !v43 )
      {
        v33 = (v43 + 8) | 4;
        v44 = v33;
      }
      else
      {
        v44 = v43 + 144;
        v33 = v44;
      }
      v32 = (__int64)v115;
      if ( v115 != (__int64 *)v44 )
      {
        while ( 1 )
        {
          v45 = (__int64 *)(v33 & 0xFFFFFFFFFFFFFFF8LL);
          if ( (v33 & 4) != 0 )
            break;
          if ( v38 == (unsigned __int8 *)v45[17] )
            goto LABEL_85;
          if ( v45 )
          {
            v33 = (__int64)(v45 + 18);
            if ( v115 == v45 + 18 )
              goto LABEL_85;
          }
          else
          {
LABEL_80:
            v33 = (unsigned __int64)(v45 + 1) | 4;
            if ( v115 == (__int64 *)v33 )
              goto LABEL_85;
          }
        }
        v30 = *v45;
        if ( v38 == *(unsigned __int8 **)(*v45 + 136) )
          goto LABEL_85;
        goto LABEL_80;
      }
LABEL_85:
      if ( i != (__int64 *)v121 )
        _libc_free(i, v29);
      if ( !v37 )
      {
        v46 = v38;
        v47 = 0;
        a1 = v46;
        goto LABEL_89;
      }
    }
    result = (__int64 *)v38;
    v47 = v37;
    a1 = (unsigned __int8 *)result;
    if ( v36 )
    {
LABEL_89:
      v48 = sub_E3D320(v47, (__int64)v29, v32, v30, v31);
      v49 = a1;
      v50 = v48;
      sub_B13360(v106, a1, v36, 0);
      v51 = (__int64)(*(_QWORD *)(v50 + 24) - *(_QWORD *)(v50 + 16)) >> 3;
      if ( (_DWORD)v117 )
      {
        v49 = (unsigned __int8 *)v106;
        if ( *(_BYTE *)(v106 + 64) )
        {
          if ( (unsigned int)v51 <= 0x80 )
          {
            v54 = sub_B12A30(v106);
            if ( (unsigned int)v117 + (unsigned __int64)v54 <= 0x10 )
            {
              v49 = (unsigned __int8 *)v116;
              sub_B12C60(v106, v116, (unsigned int)v117, v50);
              goto LABEL_92;
            }
          }
        }
      }
      else if ( (unsigned int)v51 <= 0x80 )
      {
        sub_B11F20(&i, v50);
        v78 = *(_QWORD *)(v106 + 80);
        if ( v78 )
          sub_B91220(v99, v78);
        v49 = (unsigned __int8 *)i;
        *(_QWORD *)(v106 + 80) = i;
        if ( v49 )
          sub_B976B0((__int64)&i, v49, v99);
        goto LABEL_92;
      }
      sub_B13710(v106);
LABEL_92:
      if ( v116 != (_QWORD *)v118 )
        _libc_free(v116, v49);
      v107 = 1;
      goto LABEL_53;
    }
  }
LABEL_123:
  if ( v116 != (_QWORD *)v118 )
    result = (__int64 *)_libc_free(v116, v29);
LABEL_125:
  if ( !v107 )
  {
    if ( v100 != a2 )
    {
LABEL_127:
      v55 = a2;
      do
      {
        v56 = *v55++;
        result = (__int64 *)sub_F507F0(v56);
      }
      while ( v100 != v55 );
    }
    if ( a4 != v102 )
    {
      v57 = a4;
      do
      {
        v58 = *v57++;
        result = sub_B13710(v58);
      }
      while ( v102 != v57 );
    }
  }
  return result;
}
