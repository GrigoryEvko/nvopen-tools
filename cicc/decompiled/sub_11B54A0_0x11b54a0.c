// Function: sub_11B54A0
// Address: 0x11b54a0
//
__int64 __fastcall sub_11B54A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  __int64 v8; // rbx
  unsigned int v9; // r13d
  char v10; // al
  __int64 v11; // rax
  int i; // r14d
  __int64 result; // rax
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // r15
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // r14
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 v23; // rcx
  __int64 v24; // rax
  _QWORD *v25; // rax
  __int64 v26; // rax
  _DWORD *v27; // rsi
  _DWORD *v28; // rsi
  __int64 v29; // r9
  __int64 v30; // rsi
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rax
  int v34; // r14d
  __int64 v35; // rax
  int v36; // r12d
  int v37; // r15d
  int v38; // r14d
  unsigned __int64 v39; // rdx
  int v40; // r13d
  unsigned int v41; // r14d
  __int64 v42; // r8
  __int64 v43; // r10
  __int64 v44; // r11
  int v45; // ebx
  __int64 v46; // rax
  __int64 v47; // rcx
  __int64 v48; // rax
  __int64 v49; // rsi
  __int64 v50; // rax
  _BYTE *v51; // r14
  __int64 v52; // rax
  _QWORD *v53; // rax
  __int64 v54; // r9
  __int64 v55; // r10
  _BYTE *v56; // r11
  __int64 *v57; // rsi
  _BYTE *v58; // r11
  __int64 v59; // r10
  __int64 v60; // r15
  __int64 v61; // r12
  __int64 v62; // r14
  _QWORD *v63; // rax
  _QWORD *v64; // r13
  __int64 v65; // rsi
  __int64 v66; // rsi
  __int64 v67; // rdx
  unsigned __int8 *v68; // rsi
  __int64 v69; // rax
  __int64 v70; // rdx
  __int64 v71; // rcx
  __int64 v72; // r8
  __int64 v73; // r9
  _BYTE *v74; // r11
  __int64 v75; // rax
  __int16 v76; // dx
  __int64 v77; // r15
  __int64 v78; // rsi
  __int64 v79; // r10
  _BYTE *v80; // r11
  __int64 v81; // rsi
  __int64 v82; // rdx
  unsigned __int8 *v83; // rsi
  __int64 v84; // rdx
  __int64 v85; // rdx
  __int64 v86; // rcx
  __int64 v87; // r8
  __int64 v88; // r9
  __int64 v89; // [rsp+8h] [rbp-F8h]
  __int64 v90; // [rsp+8h] [rbp-F8h]
  _BYTE *v91; // [rsp+10h] [rbp-F0h]
  _BYTE *v92; // [rsp+10h] [rbp-F0h]
  void *v93; // [rsp+18h] [rbp-E8h]
  _BYTE *v94; // [rsp+18h] [rbp-E8h]
  __int64 v95; // [rsp+18h] [rbp-E8h]
  _BYTE *v96; // [rsp+18h] [rbp-E8h]
  _BYTE *v97; // [rsp+18h] [rbp-E8h]
  __int64 v99; // [rsp+20h] [rbp-E0h]
  __int64 v100; // [rsp+20h] [rbp-E0h]
  __int64 v101; // [rsp+20h] [rbp-E0h]
  __int64 v102; // [rsp+20h] [rbp-E0h]
  __int64 v103; // [rsp+20h] [rbp-E0h]
  __int64 v104; // [rsp+20h] [rbp-E0h]
  __int64 v105; // [rsp+28h] [rbp-D8h]
  __int64 v106; // [rsp+28h] [rbp-D8h]
  __int64 v107; // [rsp+28h] [rbp-D8h]
  __int64 v108; // [rsp+28h] [rbp-D8h]
  _BYTE *v109; // [rsp+28h] [rbp-D8h]
  unsigned int v110; // [rsp+28h] [rbp-D8h]
  char v111; // [rsp+28h] [rbp-D8h]
  __int64 v112; // [rsp+30h] [rbp-D0h]
  __int64 v113; // [rsp+30h] [rbp-D0h]
  __int64 v114; // [rsp+30h] [rbp-D0h]
  __int64 v115; // [rsp+30h] [rbp-D0h]
  __int64 v116; // [rsp+30h] [rbp-D0h]
  __int64 v117; // [rsp+30h] [rbp-D0h]
  unsigned __int8 v118; // [rsp+30h] [rbp-D0h]
  unsigned int v119; // [rsp+38h] [rbp-C8h]
  __int64 v120; // [rsp+38h] [rbp-C8h]
  __int64 v121; // [rsp+38h] [rbp-C8h]
  __int64 j; // [rsp+38h] [rbp-C8h]
  int v123; // [rsp+40h] [rbp-C0h]
  unsigned int v124; // [rsp+48h] [rbp-B8h]
  _QWORD *v125; // [rsp+48h] [rbp-B8h]
  __int64 v126[4]; // [rsp+50h] [rbp-B0h] BYREF
  __int16 v127; // [rsp+70h] [rbp-90h]
  _BYTE *v128; // [rsp+80h] [rbp-80h] BYREF
  __int64 v129; // [rsp+88h] [rbp-78h]
  _BYTE v130[112]; // [rsp+90h] [rbp-70h] BYREF

  v7 = a1;
  v8 = a2;
  v9 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 32LL);
  v10 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 == 13 )
  {
    if ( *(_DWORD *)(a2 + 12) < v9 )
    {
      v27 = (_DWORD *)(a2 + 16);
      *(v27 - 2) = 0;
      sub_C8D5F0(v8, v27, v9, 4u, a5, a6);
      memset(*(void **)v8, 255, 4LL * v9);
      *(_DWORD *)(v8 + 8) = v9;
    }
    else
    {
      v14 = *(unsigned int *)(a2 + 8);
      v15 = v14;
      if ( v9 <= v14 )
        v15 = v9;
      if ( v15 )
      {
        memset(*(void **)a2, 255, 4 * v15);
        v14 = *(unsigned int *)(a2 + 8);
      }
      if ( v9 > v14 )
      {
        v16 = v9 - v14;
        if ( v16 )
        {
          if ( 4 * v16 )
            memset((void *)(*(_QWORD *)a2 + 4 * v14), 255, 4 * v16);
        }
      }
      *(_DWORD *)(a2 + 8) = v9;
    }
    if ( a3 )
      return sub_ACADE0(*(__int64 ***)(a3 + 8));
    return v7;
  }
  if ( v10 == 14 )
  {
    if ( *(_DWORD *)(a2 + 12) < v9 )
    {
      v28 = (_DWORD *)(a2 + 16);
      *(v28 - 2) = 0;
      sub_C8D5F0(v8, v28, v9, 4u, a5, a6);
      memset(*(void **)v8, 0, 4LL * v9);
      *(_DWORD *)(v8 + 8) = v9;
    }
    else
    {
      v17 = *(unsigned int *)(a2 + 8);
      v18 = v17;
      if ( v9 <= v17 )
        v18 = v9;
      if ( v18 )
      {
        memset(*(void **)a2, 0, 4 * v18);
        v17 = *(unsigned int *)(a2 + 8);
      }
      if ( v9 > v17 )
      {
        v19 = v9 - v17;
        if ( v19 )
        {
          if ( 4 * v19 )
            memset((void *)(*(_QWORD *)a2 + 4 * v17), 0, 4 * v19);
        }
      }
      *(_DWORD *)(a2 + 8) = v9;
    }
    return v7;
  }
  if ( v10 != 91 )
    goto LABEL_4;
  v20 = *(_QWORD *)(a1 - 64);
  if ( *(_BYTE *)v20 != 90 )
    goto LABEL_4;
  v21 = *(_QWORD *)(v20 - 32);
  if ( *(_BYTE *)v21 != 17 )
    goto LABEL_4;
  v23 = *(_QWORD *)(a1 - 32);
  if ( *(_BYTE *)v23 != 17 )
    goto LABEL_4;
  if ( *(_DWORD *)(v21 + 32) <= 0x40u )
    v24 = *(_QWORD *)(v21 + 24);
  else
    v24 = **(_QWORD **)(v21 + 24);
  v123 = v24;
  v25 = *(_QWORD **)(v23 + 24);
  if ( *(_DWORD *)(v23 + 32) > 0x40u )
    v25 = (_QWORD *)*v25;
  a6 = *(_QWORD *)(v20 - 64);
  v119 = (unsigned int)v25;
  if ( a3 != a6 && a3 )
  {
    v26 = *(_QWORD *)(a6 + 8);
    if ( a3 == *(_QWORD *)(a1 - 96) )
    {
      v34 = *(_DWORD *)(v26 + 32);
      if ( v9 )
      {
        v35 = *(unsigned int *)(a2 + 8);
        v113 = *(_QWORD *)(a1 - 64);
        v36 = 0;
        v37 = v34;
        v38 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 32LL);
        do
        {
          v39 = v35 + 1;
          v40 = v37 + v36;
          if ( v119 == v36 )
            v40 = v123;
          if ( v39 > *(unsigned int *)(a2 + 12) )
          {
            sub_C8D5F0(a2, (const void *)(a2 + 16), v39, 4u, a5, a6);
            v35 = *(unsigned int *)(a2 + 8);
          }
          ++v36;
          *(_DWORD *)(*(_QWORD *)a2 + 4 * v35) = v40;
          v35 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
          *(_DWORD *)(a2 + 8) = v35;
        }
        while ( v38 != v36 );
        return *(_QWORD *)(v113 - 64);
      }
      return a6;
    }
    if ( *(_QWORD *)(a3 + 8) == v26 && (unsigned __int8)sub_11AF9F0(a1, a6, a3, a2, a5, a6) )
      return *(_QWORD *)(v20 - 64);
LABEL_4:
    if ( v9 )
    {
      v11 = *(unsigned int *)(a2 + 8);
      for ( i = 0; i != v9; ++i )
      {
        if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        {
          sub_C8D5F0(a2, (const void *)(a2 + 16), v11 + 1, 4u, a5, a6);
          v11 = *(unsigned int *)(a2 + 8);
        }
        *(_DWORD *)(*(_QWORD *)a2 + 4 * v11) = i;
        v11 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
        *(_DWORD *)(a2 + 8) = v11;
      }
    }
    return v7;
  }
  v112 = *(_QWORD *)(v20 - 64);
  v105 = a4;
  result = sub_11B54A0(*(_QWORD *)(a1 - 96), a2, a6, a4, a5);
  v29 = v112;
  v30 = *(_QWORD *)(v112 + 8);
  if ( *(_QWORD *)(result + 8) == v30 )
  {
    *(_DWORD *)(*(_QWORD *)v8 + 4LL * (v119 % v9)) = *(_DWORD *)(v30 + 32) + v123;
    return result;
  }
  v31 = *(_QWORD *)(a1 + 8);
  v32 = *(_QWORD *)(*(_QWORD *)(v20 - 64) + 8LL);
  if ( *(_QWORD *)(v31 + 24) != *(_QWORD *)(v32 + 24) )
    goto LABEL_49;
  v41 = *(_DWORD *)(v32 + 32);
  v124 = *(_DWORD *)(v31 + 32);
  if ( v124 <= v41 )
    goto LABEL_49;
  v42 = 0;
  v43 = v105;
  v44 = a5;
  v128 = v130;
  v129 = 0x1000000000LL;
  if ( v41 )
  {
    v120 = v8;
    v45 = 0;
    while ( 1 )
    {
      *(_DWORD *)&v128[4 * (unsigned int)v129] = v45++;
      v46 = (unsigned int)(v129 + 1);
      LODWORD(v129) = v129 + 1;
      if ( v41 == v45 )
        break;
      if ( v46 + 1 > (unsigned __int64)HIDWORD(v129) )
      {
        v99 = v44;
        v106 = v43;
        sub_C8D5F0((__int64)&v128, v130, v46 + 1, 4u, v42, v29);
        v43 = v106;
        v44 = v99;
      }
    }
    v8 = v120;
  }
  LODWORD(v47) = v129;
  do
  {
    v48 = (unsigned int)v47;
    if ( (unsigned __int64)(unsigned int)v47 + 1 > HIDWORD(v129) )
    {
      v107 = v44;
      v114 = v43;
      sub_C8D5F0((__int64)&v128, v130, (unsigned int)v47 + 1LL, 4u, v42, v29);
      v48 = (unsigned int)v129;
      v44 = v107;
      v43 = v114;
    }
    ++v41;
    *(_DWORD *)&v128[4 * v48] = -1;
    v47 = (unsigned int)(v129 + 1);
    LODWORD(v129) = v129 + 1;
  }
  while ( v124 != v41 );
  v49 = *(_QWORD *)(v20 - 64);
  v121 = v49;
  if ( *(_BYTE *)v49 <= 0x1Cu )
  {
    v51 = 0;
    goto LABEL_117;
  }
  if ( *(_BYTE *)v49 == 84 )
  {
    v51 = *(_BYTE **)(v20 - 64);
LABEL_117:
    v50 = *(_QWORD *)(v20 + 40);
    goto LABEL_77;
  }
  v50 = *(_QWORD *)(v49 + 40);
  v51 = *(_BYTE **)(v20 - 64);
LABEL_77:
  if ( *(_QWORD *)(a1 + 40) != v50
    || (v52 = *(_QWORD *)(a1 + 16)) != 0 && !*(_QWORD *)(v52 + 8) && **(_BYTE **)(v52 + 24) == 91 )
  {
    if ( v128 != v130 )
      _libc_free(v128, v49);
    goto LABEL_49;
  }
  v108 = v44;
  v115 = v43;
  v93 = v128;
  v100 = v47;
  v127 = 257;
  v53 = sub_BD2C40(112, unk_3F1FE60);
  v55 = v115;
  v56 = (_BYTE *)v108;
  v125 = v53;
  if ( v53 )
  {
    sub_B4EB40((__int64)v53, v49, v93, v100, (__int64)v126, v54, 0);
    v56 = (_BYTE *)v108;
    v55 = v115;
  }
  if ( !v51 || *v51 == 84 )
  {
    v94 = v56;
    v101 = v55;
    v75 = sub_AA5190(*(_QWORD *)(v20 + 40));
    v77 = v75;
    if ( !v75 )
      BUG();
    v78 = *(_QWORD *)(v75 + 24);
    v118 = v76;
    v111 = HIBYTE(v76);
    v79 = v101;
    v126[0] = v78;
    v80 = v94;
    if ( v78 )
    {
      sub_B96E90((__int64)v126, v78, 1);
      v79 = v101;
      v80 = v94;
      v81 = v125[6];
      v82 = (__int64)(v125 + 6);
      if ( !v81 )
        goto LABEL_111;
    }
    else
    {
      v81 = v125[6];
      v82 = (__int64)(v125 + 6);
      if ( !v81 )
      {
LABEL_113:
        v84 = v118;
        v97 = v80;
        v104 = v79;
        BYTE1(v84) = v111;
        sub_B44220(v125, v77, v84);
        v57 = v126;
        v126[0] = (__int64)v125;
        sub_11B4E60(*(_QWORD *)(v104 + 40) + 2096LL, v126, v85, v86, v87, v88);
        v59 = v104;
        v58 = v97;
        goto LABEL_86;
      }
    }
    v92 = v80;
    v95 = v79;
    v102 = v82;
    sub_B91220(v82, v81);
    v80 = v92;
    v79 = v95;
    v82 = v102;
LABEL_111:
    v83 = (unsigned __int8 *)v126[0];
    v125[6] = v126[0];
    if ( v83 )
    {
      v96 = v80;
      v103 = v79;
      sub_B976B0((__int64)v126, v83, v82);
      v79 = v103;
      v80 = v96;
    }
    goto LABEL_113;
  }
  v57 = (__int64 *)(v51 + 24);
  v109 = v56;
  v116 = v55;
  sub_B43E90((__int64)v125, (__int64)(v51 + 24));
  v58 = v109;
  v59 = v116;
LABEL_86:
  v60 = v59;
  v61 = v89;
  v110 = v9;
  v91 = v58;
  for ( j = *(_QWORD *)(v121 + 16); j; j = *(_QWORD *)(j + 8) )
  {
    v62 = *(_QWORD *)(j + 24);
    if ( *(_BYTE *)v62 == 90 && *(_QWORD *)(v62 + 40) == v125[5] )
    {
      v127 = 257;
      v117 = *(_QWORD *)(v62 - 32);
      v63 = sub_BD2C40(72, 2u);
      v64 = v63;
      if ( v63 )
        sub_B4DE80((__int64)v63, (__int64)v125, v117, (__int64)v126, 0, 0);
      v65 = *(_QWORD *)(v62 + 48);
      v126[0] = v65;
      if ( v65 )
      {
        sub_B96E90((__int64)v126, v65, 1);
        v66 = v64[6];
        v67 = (__int64)(v64 + 6);
        if ( v66 )
          goto LABEL_95;
LABEL_96:
        v68 = (unsigned __int8 *)v126[0];
        v64[6] = v126[0];
        if ( v68 )
          sub_B976B0((__int64)v126, v68, v67);
      }
      else
      {
        v66 = v64[6];
        v67 = (__int64)(v64 + 6);
        if ( v66 )
        {
LABEL_95:
          v90 = v67;
          sub_B91220(v67, v66);
          v67 = v90;
          goto LABEL_96;
        }
      }
      LOWORD(v61) = 0;
      sub_B44220(v64, v62 + 24, v61);
      v69 = *(_QWORD *)(v60 + 40);
      v126[0] = (__int64)v64;
      sub_11B4E60(v69 + 2096, v126, v70, v71, v72, v73);
      sub_F162A0(v60, v62, (__int64)v64);
      v57 = (__int64 *)v62;
      sub_F15FC0(*(_QWORD *)(v60 + 40), v62);
      continue;
    }
  }
  v9 = v110;
  v7 = a1;
  v74 = v91;
  if ( v128 != v130 )
  {
    _libc_free(v128, v57);
    v74 = v91;
  }
  *v74 = 1;
LABEL_49:
  v33 = 0;
  if ( v9 )
  {
    do
    {
      *(_DWORD *)(*(_QWORD *)v8 + 4 * v33) = v33;
      ++v33;
    }
    while ( v33 != v9 );
  }
  return v7;
}
