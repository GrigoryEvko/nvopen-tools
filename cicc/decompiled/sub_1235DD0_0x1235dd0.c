// Function: sub_1235DD0
// Address: 0x1235dd0
//
__int64 __fastcall sub_1235DD0(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 **v3; // rbx
  __int64 **v4; // rdx
  __int64 *v5; // rdx
  _DWORD *v6; // r13
  __int64 v7; // rsi
  _QWORD *v8; // rbx
  _QWORD *v9; // r12
  __int64 v10; // rdi
  void *v11; // r14
  unsigned __int64 v13; // rax
  bool v14; // zf
  __int64 v15; // rsi
  void **v16; // r13
  void **v17; // rbx
  __int64 *v18; // rdi
  int v19; // r9d
  __int64 v20; // rax
  __int64 v21; // r8
  __int64 v22; // rdx
  __int64 *v23; // rbx
  unsigned __int64 *v24; // r12
  __int64 v25; // rsi
  __int64 v26; // r9
  __int64 v27; // rdx
  __int64 v28; // rdx
  unsigned __int64 v29; // r9
  unsigned __int64 v30; // r10
  unsigned __int64 v31; // rax
  int v32; // esi
  _BYTE *i; // rdx
  __int64 v34; // rcx
  __int64 v35; // rsi
  _QWORD *v36; // rax
  unsigned __int64 *v37; // rax
  __int64 v38; // [rsp+10h] [rbp-660h]
  int v39; // [rsp+24h] [rbp-64Ch]
  unsigned __int64 v40; // [rsp+28h] [rbp-648h]
  __int64 v41; // [rsp+30h] [rbp-640h]
  unsigned __int64 v42; // [rsp+30h] [rbp-640h]
  __int64 v43; // [rsp+30h] [rbp-640h]
  __int64 v44; // [rsp+38h] [rbp-638h]
  unsigned int v45; // [rsp+40h] [rbp-630h]
  unsigned __int64 *v46; // [rsp+48h] [rbp-628h]
  __int64 v47; // [rsp+48h] [rbp-628h]
  unsigned __int64 v48; // [rsp+50h] [rbp-620h]
  unsigned __int64 v50; // [rsp+68h] [rbp-608h]
  __int64 v51; // [rsp+68h] [rbp-608h]
  __int64 v52; // [rsp+70h] [rbp-600h]
  __int64 *v53; // [rsp+78h] [rbp-5F8h]
  char *v54; // [rsp+78h] [rbp-5F8h]
  __int64 v55; // [rsp+78h] [rbp-5F8h]
  __int64 **v57; // [rsp+80h] [rbp-5F0h]
  unsigned __int64 v58; // [rsp+80h] [rbp-5F0h]
  __int64 *v59; // [rsp+80h] [rbp-5F0h]
  unsigned __int64 v60; // [rsp+90h] [rbp-5E0h]
  __int64 v61; // [rsp+90h] [rbp-5E0h]
  unsigned __int8 v62; // [rsp+B8h] [rbp-5B8h]
  int v63; // [rsp+C4h] [rbp-5ACh] BYREF
  unsigned __int64 v64; // [rsp+C8h] [rbp-5A8h] BYREF
  __int64 *v65; // [rsp+D0h] [rbp-5A0h] BYREF
  __int64 v66; // [rsp+D8h] [rbp-598h] BYREF
  unsigned __int64 v67; // [rsp+E0h] [rbp-590h] BYREF
  __int64 v68; // [rsp+E8h] [rbp-588h] BYREF
  char *v69[2]; // [rsp+F0h] [rbp-580h] BYREF
  __int64 v70; // [rsp+100h] [rbp-570h]
  __int64 v71[4]; // [rsp+110h] [rbp-560h] BYREF
  __m128i v72[2]; // [rsp+130h] [rbp-540h] BYREF
  __m128i v73[2]; // [rsp+150h] [rbp-520h] BYREF
  unsigned __int64 v74[4]; // [rsp+170h] [rbp-500h] BYREF
  __int16 v75; // [rsp+190h] [rbp-4E0h]
  __int64 *v76; // [rsp+1A0h] [rbp-4D0h] BYREF
  __int64 v77; // [rsp+1A8h] [rbp-4C8h]
  _BYTE v78[64]; // [rsp+1B0h] [rbp-4C0h] BYREF
  const char *v79; // [rsp+1F0h] [rbp-480h] BYREF
  __int64 v80; // [rsp+1F8h] [rbp-478h]
  _BYTE v81[64]; // [rsp+200h] [rbp-470h] BYREF
  __int64 *v82; // [rsp+240h] [rbp-430h] BYREF
  _BYTE *v83; // [rsp+248h] [rbp-428h]
  __int64 v84; // [rsp+250h] [rbp-420h]
  _BYTE v85[72]; // [rsp+258h] [rbp-418h] BYREF
  __int64 *v86; // [rsp+2A0h] [rbp-3D0h] BYREF
  _BYTE *v87; // [rsp+2A8h] [rbp-3C8h]
  __int64 v88; // [rsp+2B0h] [rbp-3C0h]
  _BYTE v89[72]; // [rsp+2B8h] [rbp-3B8h] BYREF
  _BYTE *v90; // [rsp+300h] [rbp-370h] BYREF
  __int64 v91; // [rsp+308h] [rbp-368h]
  _BYTE v92[112]; // [rsp+310h] [rbp-360h] BYREF
  const char *v93; // [rsp+380h] [rbp-2F0h] BYREF
  __int64 v94; // [rsp+388h] [rbp-2E8h]
  _BYTE v95[128]; // [rsp+390h] [rbp-2E0h] BYREF
  unsigned int v96; // [rsp+410h] [rbp-260h] BYREF
  __int64 v97; // [rsp+418h] [rbp-258h]
  unsigned __int64 v98; // [rsp+428h] [rbp-248h]
  _QWORD *v99; // [rsp+430h] [rbp-240h]
  __int64 v100; // [rsp+438h] [rbp-238h]
  _QWORD v101[2]; // [rsp+440h] [rbp-230h] BYREF
  _QWORD *v102; // [rsp+450h] [rbp-220h]
  __int64 v103; // [rsp+458h] [rbp-218h]
  _QWORD v104[2]; // [rsp+460h] [rbp-210h] BYREF
  __int64 v105; // [rsp+470h] [rbp-200h]
  unsigned int v106; // [rsp+478h] [rbp-1F8h]
  char v107; // [rsp+47Ch] [rbp-1F4h]
  void *v108; // [rsp+480h] [rbp-1F0h] BYREF
  void **v109; // [rsp+488h] [rbp-1E8h]
  __int64 v110; // [rsp+4A0h] [rbp-1D0h]
  char v111; // [rsp+4A8h] [rbp-1C8h]
  unsigned __int64 *v112; // [rsp+4B0h] [rbp-1C0h] BYREF
  __int64 v113; // [rsp+4B8h] [rbp-1B8h]
  _BYTE v114[432]; // [rsp+4C0h] [rbp-1B0h] BYREF

  v3 = (__int64 **)a1;
  v4 = *(__int64 ***)(a1 + 344);
  v50 = *(_QWORD *)(a1 + 232);
  v82 = *v4;
  v83 = v85;
  v84 = 0x800000000LL;
  v5 = *v4;
  v88 = 0x800000000LL;
  v87 = v89;
  v86 = v5;
  v69[0] = 0;
  v69[1] = 0;
  v70 = 0;
  v64 = 0;
  v65 = 0;
  v96 = 0;
  v97 = 0;
  v98 = 0;
  v99 = v101;
  v100 = 0;
  LOBYTE(v101[0]) = 0;
  v102 = v104;
  v103 = 0;
  LOBYTE(v104[0]) = 0;
  v106 = 1;
  v105 = 0;
  v107 = 0;
  v6 = sub_C33320();
  sub_C3B1B0((__int64)&v112, 0.0);
  sub_C407B0(&v108, (__int64 *)&v112, v6);
  sub_C338F0((__int64)&v112);
  v7 = (__int64)&v63;
  v113 = 0x1000000000LL;
  v110 = 0;
  v111 = 0;
  v112 = (unsigned __int64 *)v114;
  v90 = v92;
  v91 = 0x200000000LL;
  if ( (unsigned __int8)sub_120C5E0(a1, &v63) )
    goto LABEL_2;
  v7 = (__int64)&v82;
  if ( (unsigned __int8)sub_1218580(a1, &v82, 0) )
    goto LABEL_2;
  v13 = *(_QWORD *)(a1 + 232);
  v7 = (__int64)&v65;
  v95[17] = 1;
  v48 = v13;
  v93 = "expected type";
  v95[16] = 3;
  if ( (unsigned __int8)sub_12190A0(a1, &v65, (int *)&v93, 1) )
    goto LABEL_2;
  v7 = (__int64)&v96;
  if ( (unsigned __int8)sub_1221570((_QWORD **)a1, (__int64)&v96, (__int64)a3, 0)
    || (v7 = (__int64)&v112, (unsigned __int8)sub_122F150(a1, (__int64)&v112, a3, 0, 0))
    || (v7 = (__int64)&v86, (unsigned __int8)sub_1218010(a1, &v86, (__int64)v69, 0, &v64))
    || (v7 = (__int64)&v90, (unsigned __int8)sub_122F1C0(a1, (__int64)&v90, a3))
    || (v7 = 56, (unsigned __int8)sub_120AFE0(a1, 56, "expected 'to' in callbr"))
    || (v7 = (__int64)&v66, v93 = 0, (unsigned __int8)sub_122FEA0(a1, &v66, (unsigned __int64 *)&v93, a3))
    || (v7 = 6, (unsigned __int8)sub_120AFE0(a1, 6, "expected '[' in callbr")) )
  {
LABEL_2:
    v62 = 1;
    goto LABEL_3;
  }
  v14 = *(_DWORD *)(a1 + 240) == 7;
  v93 = v95;
  v94 = 0x1000000000LL;
  if ( !v14 )
  {
    v7 = (__int64)&v76;
    v79 = 0;
    if ( (unsigned __int8)sub_122FEA0(a1, &v76, (unsigned __int64 *)&v79, a3) )
      goto LABEL_58;
    while ( 1 )
    {
      sub_B1A4E0((__int64)&v93, (__int64)v76);
      if ( *(_DWORD *)(a1 + 240) != 4 || !(unsigned __int8)sub_1205540(a1) )
        break;
      v79 = 0;
      v7 = (__int64)&v76;
      if ( (unsigned __int8)sub_122FEA0(a1, &v76, (unsigned __int64 *)&v79, a3) )
        goto LABEL_58;
    }
  }
  v7 = 7;
  if ( !(unsigned __int8)sub_120AFE0(a1, 7, "expected ']' at end of block list") )
  {
    v62 = sub_12104A0(a1, v65, (__int64)v112, (unsigned int)v113, &v67);
    if ( v62 )
    {
      v7 = v48;
      v81[17] = 1;
      v79 = "Invalid result type for LLVM function";
      v81[16] = 3;
      sub_11FD800(a1 + 176, v48, (__int64)&v79, 1);
      goto LABEL_47;
    }
    v18 = *(__int64 **)a1;
    v98 = v67;
    v7 = sub_BCE3C0(v18, 0);
    v62 = sub_121E800(v3, v7, &v96, &v68, a3, v19);
    if ( !v62 )
    {
      v76 = (__int64 *)v78;
      v79 = v81;
      v77 = 0x800000000LL;
      v80 = 0x800000000LL;
      v20 = *(_QWORD *)(v67 + 16);
      v21 = v20 + 8;
      v53 = (__int64 *)(v20 + 8LL * *(unsigned int *)(v67 + 12));
      v22 = 3LL * (unsigned int)v113;
      v46 = &v112[v22];
      if ( &v112[v22] != v112 )
      {
        v57 = v3;
        v23 = (__int64 *)(v20 + 8);
        v24 = v112;
        while ( 1 )
        {
          if ( v53 == v23 )
          {
            if ( !(*(_DWORD *)(v67 + 8) >> 8) )
            {
              v74[0] = (unsigned __int64)"too many arguments specified";
              v75 = 259;
              v7 = *v24;
              sub_11FD800((__int64)(v57 + 22), *v24, (__int64)v74, 1);
              v62 = 1;
              goto LABEL_76;
            }
            v26 = v24[1];
          }
          else
          {
            v25 = *v23;
            if ( *v23 )
            {
              v26 = v24[1];
              if ( v25 != *(_QWORD *)(v26 + 8) )
              {
                sub_1207630(v71, v25);
                sub_95D570(v72, "argument is not of expected type '", (__int64)v71);
                sub_94F930(v73, (__int64)v72, "'");
                v75 = 260;
                v74[0] = (unsigned __int64)v73;
                v7 = *v24;
                sub_11FD800((__int64)(v57 + 22), *v24, (__int64)v74, 1);
                sub_2240A30(v73);
                sub_2240A30(v72);
                sub_2240A30(v71);
                v62 = 1;
                goto LABEL_76;
              }
              ++v23;
            }
            else
            {
              v26 = v24[1];
              ++v23;
            }
          }
          v27 = (unsigned int)v77;
          if ( (unsigned __int64)(unsigned int)v77 + 1 > HIDWORD(v77) )
          {
            v43 = v26;
            sub_C8D5F0((__int64)&v76, v78, (unsigned int)v77 + 1LL, 8u, v21, v26);
            v27 = (unsigned int)v77;
            v26 = v43;
          }
          v76[v27] = v26;
          v28 = (unsigned int)v80;
          LODWORD(v77) = v77 + 1;
          v29 = (unsigned int)v80 + 1LL;
          v30 = v24[2];
          if ( v29 > HIDWORD(v80) )
          {
            v42 = v24[2];
            sub_C8D5F0((__int64)&v79, v81, (unsigned int)v80 + 1LL, 8u, v21, v29);
            v28 = (unsigned int)v80;
            v30 = v42;
          }
          v24 += 3;
          *(_QWORD *)&v79[8 * v28] = v30;
          LODWORD(v80) = v80 + 1;
          if ( v46 == v24 )
          {
            v21 = (__int64)v23;
            v3 = v57;
            break;
          }
        }
      }
      if ( v53 == (__int64 *)v21 )
      {
        v54 = (char *)v79;
        v58 = (unsigned int)v80;
        v60 = sub_A7A280(*v3, (__int64)&v82);
        v31 = sub_A7A280(*v3, (__int64)&v86);
        v32 = 0;
        v40 = sub_A78180(*v3, v31, v60, v54, v58);
        v59 = v76;
        v75 = 257;
        v44 = (unsigned int)v77;
        v41 = (unsigned int)v91;
        v47 = (__int64)v93;
        v45 = v94;
        v52 = v66;
        v51 = v68;
        v55 = v67;
        for ( i = v90; &v90[56 * (unsigned int)v91] != i; i += 56 )
        {
          v34 = *((_QWORD *)i + 5) - *((_QWORD *)i + 4);
          v32 += v34 >> 3;
        }
        v38 = (__int64)v90;
        v35 = (unsigned int)(v94 + v77 + 2 + v32);
        LOBYTE(v39) = 16 * (_DWORD)v91 != 0;
        v36 = sub_BD2CC0(96, ((unsigned __int64)(unsigned int)(16 * v91) << 32) | v35);
        v61 = (__int64)v36;
        if ( v36 )
        {
          sub_B44260((__int64)v36, **(_QWORD **)(v55 + 16), 11, v35 & 0x7FFFFFF | (v39 << 28), 0, 0);
          *(_QWORD *)(v61 + 72) = 0;
          sub_B4B130(v61, v55, v51, v52, v47, v45, v59, v44, v38, v41, (__int64)v74);
        }
        *(_WORD *)(v61 + 2) = (4 * v63) | *(_WORD *)(v61 + 2) & 0xF003;
        *(_QWORD *)(v61 + 72) = v40;
        v74[0] = v61;
        v37 = sub_121BCE0(v3 + 179, v74);
        sub_1205F70((__int64)v37, v69);
        v7 = v61;
        *a2 = v61;
      }
      else
      {
        v7 = v50;
        v74[0] = (unsigned __int64)"not enough parameters specified for call";
        v75 = 259;
        sub_11FD800((__int64)(v3 + 22), v50, (__int64)v74, 1);
        v62 = 1;
      }
LABEL_76:
      if ( v79 != v81 )
        _libc_free(v79, v7);
      if ( v76 != (__int64 *)v78 )
        _libc_free(v76, v7);
      goto LABEL_47;
    }
  }
LABEL_58:
  v62 = 1;
LABEL_47:
  if ( v93 != v95 )
    _libc_free(v93, v7);
LABEL_3:
  v8 = v90;
  v9 = &v90[56 * (unsigned int)v91];
  if ( v90 != (_BYTE *)v9 )
  {
    do
    {
      v10 = *(v9 - 3);
      v9 -= 7;
      if ( v10 )
      {
        v7 = v9[6] - v10;
        j_j___libc_free_0(v10, v7);
      }
      if ( (_QWORD *)*v9 != v9 + 2 )
      {
        v7 = v9[2] + 1LL;
        j_j___libc_free_0(*v9, v7);
      }
    }
    while ( v8 != v9 );
    v9 = v90;
  }
  if ( v9 != (_QWORD *)v92 )
    _libc_free(v9, v7);
  if ( v112 != (unsigned __int64 *)v114 )
    _libc_free(v112, v7);
  if ( v110 )
    j_j___libc_free_0_0(v110);
  v11 = sub_C33340();
  if ( v108 == v11 )
  {
    if ( v109 )
    {
      v15 = 3LL * (_QWORD)*(v109 - 1);
      v16 = &v109[v15];
      if ( v109 == &v109[v15] )
      {
        v7 = v15 * 8 + 8;
        j_j_j___libc_free_0_0(v16 - 1);
      }
      else
      {
        do
        {
          while ( 1 )
          {
            v17 = v16;
            v16 -= 3;
            if ( v11 == *v16 )
              break;
            sub_C338F0((__int64)v16);
            if ( v109 == v16 )
              goto LABEL_55;
          }
          sub_969EE0((__int64)v16);
        }
        while ( v109 != v16 );
LABEL_55:
        v7 = 24LL * (_QWORD)*(v17 - 4) + 8;
        j_j_j___libc_free_0_0(v16 - 1);
      }
    }
  }
  else
  {
    sub_C338F0((__int64)&v108);
  }
  if ( v106 > 0x40 && v105 )
    j_j___libc_free_0_0(v105);
  if ( v102 != v104 )
  {
    v7 = v104[0] + 1LL;
    j_j___libc_free_0(v102, v104[0] + 1LL);
  }
  if ( v99 != v101 )
  {
    v7 = v101[0] + 1LL;
    j_j___libc_free_0(v99, v101[0] + 1LL);
  }
  if ( v69[0] )
  {
    v7 = v70 - (unsigned __int64)v69[0];
    j_j___libc_free_0(v69[0], v70 - (unsigned __int64)v69[0]);
  }
  if ( v87 != v89 )
    _libc_free(v87, v7);
  if ( v83 != v85 )
    _libc_free(v83, v7);
  return v62;
}
