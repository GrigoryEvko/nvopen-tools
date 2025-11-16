// Function: sub_1236B50
// Address: 0x1236b50
//
__int64 __fastcall sub_1236B50(__int64 a1, __int64 *a2, __int64 *a3, int a4)
{
  _QWORD **v5; // rbx
  __int64 **v6; // rdx
  __int64 *v7; // rdx
  _DWORD *v8; // r12
  unsigned __int64 v9; // rax
  __int64 v10; // rsi
  int v11; // r12d
  _QWORD *v12; // r12
  _QWORD *v13; // r15
  __int64 v14; // rdi
  void *v15; // r13
  unsigned __int64 v17; // rax
  __int64 v18; // rsi
  void **v19; // r12
  void **v20; // rbx
  __int64 *v21; // rdi
  int v22; // r9d
  __int64 v23; // r9
  __int64 v24; // rax
  __int64 *v25; // r15
  __int64 v26; // rdx
  unsigned __int64 *v27; // r13
  __int64 v28; // rsi
  __int64 v29; // r8
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  __int64 v32; // rax
  unsigned __int64 v33; // rdx
  __int64 v34; // r8
  unsigned __int64 v35; // r15
  unsigned __int64 v36; // rax
  unsigned __int64 v37; // rax
  __int64 v38; // r13
  _BYTE *v39; // r11
  _BYTE *v40; // rsi
  int v41; // r8d
  __int64 v42; // rdi
  unsigned __int64 v43; // rsi
  _QWORD *v44; // rax
  __int64 v45; // r15
  __int16 v46; // ax
  unsigned __int64 *v47; // rax
  __int64 v48; // rsi
  _QWORD *v49; // r12
  _QWORD **v50; // r14
  _QWORD *i; // rbx
  __int64 v52; // rdi
  __int64 v53; // [rsp+8h] [rbp-5C8h]
  unsigned int v54; // [rsp+10h] [rbp-5C0h]
  __int64 v55; // [rsp+20h] [rbp-5B0h]
  __int64 v56; // [rsp+20h] [rbp-5B0h]
  __int64 v57; // [rsp+20h] [rbp-5B0h]
  unsigned __int64 v58; // [rsp+30h] [rbp-5A0h]
  __int64 v59; // [rsp+40h] [rbp-590h]
  unsigned __int64 v60; // [rsp+48h] [rbp-588h]
  __int64 *v61; // [rsp+50h] [rbp-580h]
  unsigned __int64 v64; // [rsp+68h] [rbp-568h]
  unsigned __int64 *v66; // [rsp+70h] [rbp-560h]
  _QWORD *v67; // [rsp+70h] [rbp-560h]
  unsigned __int8 v68; // [rsp+A8h] [rbp-528h]
  __int64 *v69; // [rsp+A8h] [rbp-528h]
  unsigned __int64 v70; // [rsp+A8h] [rbp-528h]
  __int64 v71; // [rsp+A8h] [rbp-528h]
  int v72; // [rsp+B8h] [rbp-518h] BYREF
  int v73; // [rsp+BCh] [rbp-514h] BYREF
  unsigned __int64 v74; // [rsp+C0h] [rbp-510h] BYREF
  __int64 *v75; // [rsp+C8h] [rbp-508h] BYREF
  unsigned __int64 v76; // [rsp+D0h] [rbp-500h] BYREF
  __int64 v77; // [rsp+D8h] [rbp-4F8h] BYREF
  char *v78[2]; // [rsp+E0h] [rbp-4F0h] BYREF
  __int64 v79; // [rsp+F0h] [rbp-4E0h]
  __int64 v80[4]; // [rsp+100h] [rbp-4D0h] BYREF
  __m128i v81[2]; // [rsp+120h] [rbp-4B0h] BYREF
  __m128i v82[2]; // [rsp+140h] [rbp-490h] BYREF
  unsigned __int64 v83[4]; // [rsp+160h] [rbp-470h] BYREF
  __int16 v84; // [rsp+180h] [rbp-450h]
  _BYTE *v85; // [rsp+190h] [rbp-440h] BYREF
  __int64 v86; // [rsp+198h] [rbp-438h]
  _BYTE v87[64]; // [rsp+1A0h] [rbp-430h] BYREF
  const char *v88; // [rsp+1E0h] [rbp-3F0h] BYREF
  __int64 v89; // [rsp+1E8h] [rbp-3E8h]
  _BYTE v90[16]; // [rsp+1F0h] [rbp-3E0h] BYREF
  char v91; // [rsp+200h] [rbp-3D0h]
  char v92; // [rsp+201h] [rbp-3CFh]
  __int64 *v93; // [rsp+230h] [rbp-3A0h] BYREF
  _BYTE *v94; // [rsp+238h] [rbp-398h]
  __int64 v95; // [rsp+240h] [rbp-390h]
  _BYTE v96[72]; // [rsp+248h] [rbp-388h] BYREF
  __int64 *v97; // [rsp+290h] [rbp-340h] BYREF
  _BYTE *v98; // [rsp+298h] [rbp-338h]
  __int64 v99; // [rsp+2A0h] [rbp-330h]
  _BYTE v100[72]; // [rsp+2A8h] [rbp-328h] BYREF
  _BYTE *v101; // [rsp+2F0h] [rbp-2E0h] BYREF
  __int64 v102; // [rsp+2F8h] [rbp-2D8h]
  _BYTE v103[112]; // [rsp+300h] [rbp-2D0h] BYREF
  unsigned int v104; // [rsp+370h] [rbp-260h] BYREF
  __int64 v105; // [rsp+378h] [rbp-258h]
  unsigned __int64 v106; // [rsp+388h] [rbp-248h]
  _BYTE *v107; // [rsp+390h] [rbp-240h]
  size_t v108; // [rsp+398h] [rbp-238h]
  _QWORD v109[2]; // [rsp+3A0h] [rbp-230h] BYREF
  _QWORD *v110; // [rsp+3B0h] [rbp-220h]
  __int64 v111; // [rsp+3B8h] [rbp-218h]
  _QWORD v112[2]; // [rsp+3C0h] [rbp-210h] BYREF
  __int64 v113; // [rsp+3D0h] [rbp-200h]
  unsigned int v114; // [rsp+3D8h] [rbp-1F8h]
  char v115; // [rsp+3DCh] [rbp-1F4h]
  void *v116; // [rsp+3E0h] [rbp-1F0h] BYREF
  void **v117; // [rsp+3E8h] [rbp-1E8h]
  __int64 v118; // [rsp+400h] [rbp-1D0h]
  char v119; // [rsp+408h] [rbp-1C8h]
  unsigned __int64 *v120; // [rsp+410h] [rbp-1C0h] BYREF
  __int64 v121; // [rsp+418h] [rbp-1B8h]
  _BYTE v122[432]; // [rsp+420h] [rbp-1B0h] BYREF

  v5 = (_QWORD **)a1;
  v6 = *(__int64 ***)(a1 + 344);
  v93 = *v6;
  v94 = v96;
  v95 = 0x800000000LL;
  v7 = *v6;
  v99 = 0x800000000LL;
  v97 = v7;
  v107 = v109;
  v98 = v100;
  v78[0] = 0;
  v78[1] = 0;
  v79 = 0;
  v74 = 0;
  v75 = 0;
  v104 = 0;
  v105 = 0;
  v106 = 0;
  v108 = 0;
  LOBYTE(v109[0]) = 0;
  v110 = v112;
  v111 = 0;
  LOBYTE(v112[0]) = 0;
  v114 = 1;
  v113 = 0;
  v115 = 0;
  v8 = sub_C33320();
  sub_C3B1B0((__int64)&v120, 0.0);
  sub_C407B0(&v116, (__int64 *)&v120, v8);
  sub_C338F0((__int64)&v120);
  v118 = 0;
  v120 = (unsigned __int64 *)v122;
  v121 = 0x1000000000LL;
  v101 = v103;
  v102 = 0x200000000LL;
  v9 = *(_QWORD *)(a1 + 232);
  v119 = 0;
  v64 = v9;
  if ( a4
    && (v10 = 354, (unsigned __int8)sub_120AFE0(a1, 354, "expected 'tail call', 'musttail call', or 'notail call'")) )
  {
LABEL_14:
    v68 = 1;
  }
  else
  {
    v11 = 0;
    while ( 2 )
    {
      switch ( *(_DWORD *)(a1 + 240) )
      {
        case 'M':
          v11 |= 2u;
          *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
          continue;
        case 'N':
          v11 |= 4u;
          *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
          continue;
        case 'O':
          v11 |= 8u;
          *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
          continue;
        case 'P':
          v11 |= 0x10u;
          *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
          continue;
        case 'Q':
          v11 |= 0x20u;
          *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
          continue;
        case 'R':
          v11 |= 1u;
          *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
          continue;
        case 'S':
          v11 |= 0x40u;
          *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
          continue;
        case 'T':
          v11 = -1;
          *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
          continue;
        default:
          v10 = (__int64)&v73;
          if ( (unsigned __int8)sub_120C5E0(a1, &v73) )
            goto LABEL_14;
          v10 = (__int64)&v93;
          if ( (unsigned __int8)sub_1218580(a1, &v93, 0) )
            goto LABEL_14;
          v10 = (__int64)&v72;
          if ( (unsigned __int8)sub_1212650(a1, &v72, *(_DWORD *)(*(_QWORD *)(a1 + 344) + 320LL)) )
            goto LABEL_14;
          v17 = *(_QWORD *)(a1 + 232);
          v10 = (__int64)&v75;
          v92 = 1;
          v60 = v17;
          v88 = "expected type";
          v91 = 3;
          if ( (unsigned __int8)sub_12190A0(a1, &v75, (int *)&v88, 1) )
            goto LABEL_14;
          v10 = (__int64)&v104;
          if ( (unsigned __int8)sub_1221570((_QWORD **)a1, (__int64)&v104, (__int64)a3, 0) )
            goto LABEL_14;
          v10 = (__int64)&v120;
          if ( (unsigned __int8)sub_122F150(
                                  a1,
                                  (__int64)&v120,
                                  a3,
                                  a4 == 2,
                                  *(_DWORD *)(*(_QWORD *)(a3[1] + 24) + 8LL) >> 8 != 0) )
            goto LABEL_14;
          v10 = (__int64)&v97;
          if ( (unsigned __int8)sub_1218010(a1, &v97, (__int64)v78, 0, &v74) )
            goto LABEL_14;
          v10 = (__int64)&v101;
          if ( (unsigned __int8)sub_122F1C0(a1, (__int64)&v101, a3) )
            goto LABEL_14;
          v68 = sub_12104A0(a1, v75, (__int64)v120, (unsigned int)v121, &v76);
          if ( v68 )
          {
            v10 = v60;
            v88 = "Invalid result type for LLVM function";
            v92 = 1;
            v91 = 3;
            sub_11FD800(a1 + 176, v60, (__int64)&v88, 1);
            goto LABEL_15;
          }
          v21 = *(__int64 **)a1;
          v106 = v76;
          v10 = sub_BCE3C0(v21, v72);
          if ( (unsigned __int8)sub_121E800(v5, v10, &v104, &v77, a3, v22) )
            goto LABEL_14;
          v85 = v87;
          v88 = v90;
          v86 = 0x800000000LL;
          v89 = 0x800000000LL;
          v24 = *(_QWORD *)(v76 + 16);
          v25 = (__int64 *)(v24 + 8);
          v69 = (__int64 *)(v24 + 8LL * *(unsigned int *)(v76 + 12));
          v26 = 3LL * (unsigned int)v121;
          v66 = &v120[v26];
          if ( &v120[v26] == v120 )
            goto LABEL_74;
          v27 = v120;
          break;
      }
      break;
    }
    do
    {
      if ( v69 == v25 )
      {
        if ( !(*(_DWORD *)(v76 + 8) >> 8) )
        {
          v83[0] = (unsigned __int64)"too many arguments specified";
          v84 = 259;
          v10 = *v27;
          sub_11FD800((__int64)(v5 + 22), *v27, (__int64)v83, 1);
          v68 = 1;
          goto LABEL_76;
        }
        v29 = v27[1];
      }
      else
      {
        v28 = *v25;
        v29 = v27[1];
        if ( *v25 && v28 != *(_QWORD *)(v29 + 8) )
        {
          sub_1207630(v80, v28);
          sub_95D570(v81, "argument is not of expected type '", (__int64)v80);
          sub_94F930(v82, (__int64)v81, "'");
          v84 = 260;
          v83[0] = (unsigned __int64)v82;
          v10 = *v27;
          sub_11FD800((__int64)(v5 + 22), *v27, (__int64)v83, 1);
          sub_2240A30(v82);
          sub_2240A30(v81);
          sub_2240A30(v80);
          v68 = 1;
          goto LABEL_76;
        }
        ++v25;
      }
      v30 = (unsigned int)v89;
      v31 = (unsigned int)v89 + 1LL;
      if ( v31 > HIDWORD(v89) )
      {
        v56 = v29;
        sub_C8D5F0((__int64)&v88, v90, v31, 8u, v29, v23);
        v30 = (unsigned int)v89;
        v29 = v56;
      }
      *(_QWORD *)&v88[8 * v30] = v29;
      v32 = (unsigned int)v86;
      LODWORD(v89) = v89 + 1;
      v33 = (unsigned int)v86 + 1LL;
      v34 = v27[2];
      if ( v33 > HIDWORD(v86) )
      {
        v55 = v27[2];
        sub_C8D5F0((__int64)&v85, v87, v33, 8u, v34, v23);
        v32 = (unsigned int)v86;
        v34 = v55;
      }
      v27 += 3;
      *(_QWORD *)&v85[8 * v32] = v34;
      LODWORD(v86) = v86 + 1;
    }
    while ( v66 != v27 );
LABEL_74:
    if ( v69 != v25 )
    {
      v10 = v64;
      v83[0] = (unsigned __int64)"not enough parameters specified for call";
      v84 = 259;
      sub_11FD800((__int64)(v5 + 22), v64, (__int64)v83, 1);
      v68 = 1;
      goto LABEL_76;
    }
    v67 = v85;
    v70 = (unsigned int)v86;
    v35 = sub_A7A280(*v5, (__int64)&v93);
    v36 = sub_A7A280(*v5, (__int64)&v97);
    v37 = sub_A78180(*v5, v36, v35, v67, v70);
    v84 = 257;
    v58 = v37;
    v61 = (__int64 *)v88;
    v38 = v76;
    v59 = (unsigned int)v89;
    v39 = &v101[56 * (unsigned int)v102];
    v71 = v77;
    if ( v101 == v39 )
    {
      v41 = 0;
    }
    else
    {
      v40 = v101;
      v41 = 0;
      do
      {
        v42 = *((_QWORD *)v40 + 5) - *((_QWORD *)v40 + 4);
        v40 += 56;
        v41 += v42 >> 3;
      }
      while ( v39 != v40 );
    }
    v53 = (unsigned int)v102;
    LOBYTE(v67) = 16 * (_DWORD)v102 != 0;
    v54 = v89 + v41 + 1;
    v43 = ((unsigned __int64)(unsigned int)(16 * v102) << 32) | v54;
    v57 = (__int64)v101;
    v44 = sub_BD2CC0(88, v43);
    v45 = (__int64)v44;
    if ( v44 )
    {
      sub_B44260((__int64)v44, **(_QWORD **)(v38 + 16), 56, v54 & 0x7FFFFFF | ((_DWORD)v67 << 28), 0, 0);
      v43 = v38;
      *(_QWORD *)(v45 + 72) = 0;
      sub_B4A290(v45, v38, v71, v61, v59, (__int64)v83, v57, v53);
    }
    v46 = a4 | *(_WORD *)(v45 + 2) & 0xFFFC;
    *(_WORD *)(v45 + 2) = v46;
    *(_WORD *)(v45 + 2) = (4 * v73) | v46 & 0xF003;
    if ( v11 )
    {
      if ( (unsigned __int8)sub_920620(v45) )
      {
        sub_B45150(v45, v11);
        goto LABEL_87;
      }
      sub_BD72D0(v45, v43);
      v10 = v64;
      v83[0] = (unsigned __int64)"fast-math-flags specified for call without floating-point scalar or vector return type";
      v84 = 259;
      sub_11FD800((__int64)(v5 + 22), v64, (__int64)v83, 1);
      v68 = 1;
    }
    else
    {
LABEL_87:
      if ( v104 != 3 || (v48 = v108, !(unsigned __int8)sub_12101E0(v107, v108)) )
      {
LABEL_88:
        *(_QWORD *)(v45 + 72) = v58;
        v83[0] = v45;
        v47 = sub_121BCE0(v5 + 179, v83);
        v10 = (__int64)v78;
        sub_1205F70((__int64)v47, v78);
        v68 = 0;
        *a2 = v45;
        goto LABEL_76;
      }
      v68 = *((_BYTE *)v5 + 1745);
      if ( !v68 )
      {
        if ( !*((_BYTE *)v5 + 1746) )
        {
          v49 = v5[43];
          v50 = v5;
          for ( i = (_QWORD *)v49[4]; v49 + 3 != i; i = (_QWORD *)i[1] )
          {
            if ( i )
              v52 = (__int64)(i - 7);
            else
              v52 = 0;
            sub_B2BA20(v52, 0);
          }
          *((_BYTE *)v49 + 872) = 0;
          v5 = v50;
        }
        *((_BYTE *)v5 + 1746) = 1;
        goto LABEL_88;
      }
      sub_BD72D0(v45, v48);
      v10 = v64;
      v83[0] = (unsigned __int64)"llvm.dbg intrinsic should not appear in a module using non-intrinsic debug info";
      v84 = 259;
      sub_11FD800((__int64)(v5 + 22), v64, (__int64)v83, 1);
    }
LABEL_76:
    if ( v88 != v90 )
      _libc_free(v88, v10);
    if ( v85 != v87 )
      _libc_free(v85, v10);
  }
LABEL_15:
  v12 = v101;
  v13 = &v101[56 * (unsigned int)v102];
  if ( v101 != (_BYTE *)v13 )
  {
    do
    {
      v14 = *(v13 - 3);
      v13 -= 7;
      if ( v14 )
      {
        v10 = v13[6] - v14;
        j_j___libc_free_0(v14, v10);
      }
      if ( (_QWORD *)*v13 != v13 + 2 )
      {
        v10 = v13[2] + 1LL;
        j_j___libc_free_0(*v13, v10);
      }
    }
    while ( v12 != v13 );
    v13 = v101;
  }
  if ( v13 != (_QWORD *)v103 )
    _libc_free(v13, v10);
  if ( v120 != (unsigned __int64 *)v122 )
    _libc_free(v120, v10);
  if ( v118 )
    j_j___libc_free_0_0(v118);
  v15 = sub_C33340();
  if ( v116 == v15 )
  {
    if ( v117 )
    {
      v18 = 3LL * (_QWORD)*(v117 - 1);
      v19 = &v117[v18];
      if ( v117 == &v117[v18] )
      {
        v10 = v18 * 8 + 8;
        j_j_j___libc_free_0_0(v19 - 1);
      }
      else
      {
        do
        {
          while ( 1 )
          {
            v20 = v19;
            v19 -= 3;
            if ( v15 == *v19 )
              break;
            sub_C338F0((__int64)v19);
            if ( v117 == v19 )
              goto LABEL_59;
          }
          sub_969EE0((__int64)v19);
        }
        while ( v117 != v19 );
LABEL_59:
        v10 = 24LL * (_QWORD)*(v20 - 4) + 8;
        j_j_j___libc_free_0_0(v19 - 1);
      }
    }
  }
  else
  {
    sub_C338F0((__int64)&v116);
  }
  if ( v114 > 0x40 && v113 )
    j_j___libc_free_0_0(v113);
  if ( v110 != v112 )
  {
    v10 = v112[0] + 1LL;
    j_j___libc_free_0(v110, v112[0] + 1LL);
  }
  if ( v107 != (_BYTE *)v109 )
  {
    v10 = v109[0] + 1LL;
    j_j___libc_free_0(v107, v109[0] + 1LL);
  }
  if ( v78[0] )
  {
    v10 = v79 - (unsigned __int64)v78[0];
    j_j___libc_free_0(v78[0], v79 - (unsigned __int64)v78[0]);
  }
  if ( v98 != v100 )
    _libc_free(v98, v10);
  if ( v94 != v96 )
    _libc_free(v94, v10);
  return v68;
}
