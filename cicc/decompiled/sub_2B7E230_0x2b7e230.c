// Function: sub_2B7E230
// Address: 0x2b7e230
//
_QWORD *__fastcall sub_2B7E230(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        __int64 a6,
        __int64 (__fastcall *a7)(__int64, __int64, _QWORD *, __int64 *, _QWORD),
        __int64 a8)
{
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 v12; // rdi
  unsigned int v13; // edx
  unsigned __int64 *v14; // rax
  int v15; // eax
  __int64 v16; // r13
  __int64 v17; // r15
  __int64 v18; // rbx
  _QWORD *v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // rdx
  unsigned int v23; // eax
  __int64 v24; // rax
  int v25; // ecx
  __int64 v26; // rax
  bool v27; // zf
  _BYTE *v28; // rdx
  int v29; // eax
  __int64 **v30; // rbx
  unsigned __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  _QWORD *v34; // r14
  __int64 *v35; // rsi
  __int64 *v36; // rax
  __int64 *v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // r13
  __int64 v41; // r13
  _DWORD *v42; // rax
  _DWORD *v43; // rdx
  _QWORD *v44; // rax
  __int64 v45; // rdx
  _QWORD *v46; // rsi
  __int64 v47; // rcx
  __int64 v48; // rdx
  _QWORD *v49; // rdx
  int *v50; // r13
  int *v51; // r12
  _QWORD *v52; // rsi
  __int64 v53; // rax
  __int64 v54; // rax
  unsigned __int64 v55; // r13
  _BYTE *v56; // r12
  _QWORD *v57; // rsi
  unsigned int v58; // ecx
  __int64 v59; // rdx
  __int64 v60; // rax
  unsigned __int64 v61; // rbx
  unsigned __int64 v62; // rdi
  unsigned __int64 v64; // rdi
  char v65; // dl
  bool v66; // bl
  unsigned __int64 v67; // rbx
  _QWORD *v68; // rax
  __int64 v69; // r8
  __int64 v70; // r9
  _BYTE *v71; // r10
  unsigned __int64 v72; // rax
  int *v73; // rsi
  __int64 v74; // rax
  unsigned __int64 v75; // rdx
  int v76; // edi
  __int64 v77; // rsi
  _QWORD *v78; // rax
  _QWORD *v79; // rdx
  int v80; // eax
  _BYTE *v81; // rsi
  __int64 v82; // rax
  unsigned __int64 v83; // [rsp+20h] [rbp-1E0h]
  __int64 v84; // [rsp+30h] [rbp-1D0h]
  __int64 v86; // [rsp+40h] [rbp-1C0h]
  int v87; // [rsp+40h] [rbp-1C0h]
  __int64 v89; // [rsp+48h] [rbp-1B8h]
  int v91; // [rsp+64h] [rbp-19Ch]
  __int64 v93; // [rsp+70h] [rbp-190h] BYREF
  __int64 v94; // [rsp+78h] [rbp-188h] BYREF
  char v95[48]; // [rsp+80h] [rbp-180h] BYREF
  int *v96; // [rsp+B0h] [rbp-150h] BYREF
  __int64 v97; // [rsp+B8h] [rbp-148h]
  _BYTE v98[48]; // [rsp+C0h] [rbp-140h] BYREF
  __int64 *v99; // [rsp+F0h] [rbp-110h] BYREF
  __int64 v100; // [rsp+F8h] [rbp-108h]
  __int64 v101; // [rsp+100h] [rbp-100h] BYREF
  int v102; // [rsp+108h] [rbp-F8h]
  char v103; // [rsp+10Ch] [rbp-F4h]
  char v104; // [rsp+110h] [rbp-F0h] BYREF
  _BYTE *v105; // [rsp+130h] [rbp-D0h] BYREF
  __int64 v106; // [rsp+138h] [rbp-C8h]
  _BYTE v107[64]; // [rsp+140h] [rbp-C0h] BYREF
  _BYTE *v108; // [rsp+180h] [rbp-80h] BYREF
  __int64 v109; // [rsp+188h] [rbp-78h]
  _BYTE v110[24]; // [rsp+190h] [rbp-70h] BYREF
  int v111; // [rsp+1A8h] [rbp-58h] BYREF
  unsigned __int64 v112; // [rsp+1B0h] [rbp-50h]
  int *v113; // [rsp+1B8h] [rbp-48h]
  int *v114; // [rsp+1C0h] [rbp-40h]
  __int64 v115; // [rsp+1C8h] [rbp-38h]

  v8 = a5;
  v111 = 0;
  v112 = 0;
  v115 = 0;
  v105 = v107;
  v106 = 0x400000000LL;
  v109 = 0x400000000LL;
  v113 = &v111;
  v114 = &v111;
  v9 = *(_QWORD *)(a1 + 3312);
  v10 = *(unsigned int *)(v9 + 24);
  v108 = v110;
  v11 = *(_QWORD *)(a1 + 3416);
  v12 = *(_QWORD *)(v9 + 8);
  if ( !(_DWORD)v10 )
  {
LABEL_121:
    v86 = 0;
    goto LABEL_4;
  }
  v10 = (unsigned int)(v10 - 1);
  v13 = v10 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
  v14 = (unsigned __int64 *)(v12 + 16LL * v13);
  a5 = *v14;
  if ( v11 != *v14 )
  {
    v80 = 1;
    while ( a5 != -4096 )
    {
      a6 = (unsigned int)(v80 + 1);
      v13 = v10 & (v80 + v13);
      v14 = (unsigned __int64 *)(v12 + 16LL * v13);
      a5 = *v14;
      if ( v11 == *v14 )
        goto LABEL_3;
      v80 = a6;
    }
    goto LABEL_121;
  }
LABEL_3:
  v86 = v14[1];
LABEL_4:
  v15 = 0;
  LODWORD(v96) = 0;
  v91 = a3;
  if ( (int)a3 <= 0 )
    goto LABEL_22;
  v84 = v8;
  do
  {
    v16 = *(_QWORD *)(a2 + 8LL * v15);
    if ( *(_BYTE *)v16 <= 0x1Cu )
      goto LABEL_85;
    v17 = *(_QWORD *)(a1 + 3416);
    v18 = *(_QWORD *)(v16 + 40);
    v99 = 0;
    v100 = (__int64)&v104;
    v103 = 1;
    v101 = 4;
    v102 = 0;
    if ( v17 != 0 && v17 != v18 )
    {
LABEL_8:
      v19 = (_QWORD *)v100;
      v10 = HIDWORD(v101);
      v20 = (__int64 *)(v100 + 8LL * HIDWORD(v101));
      if ( (__int64 *)v100 != v20 )
      {
        while ( *v19 != v17 )
        {
          if ( v20 == ++v19 )
            goto LABEL_74;
        }
LABEL_12:
        if ( (*(_BYTE *)(a1 + 88) & 1) != 0 )
          goto LABEL_13;
        goto LABEL_100;
      }
LABEL_74:
      if ( HIDWORD(v101) < (unsigned int)v101 )
      {
        ++HIDWORD(v101);
        *v20 = v17;
        v99 = (__int64 *)((char *)v99 + 1);
        goto LABEL_76;
      }
      while ( 1 )
      {
        sub_C8CC70((__int64)&v99, v17, (__int64)v20, v10, a5, a6);
        v64 = v100;
        if ( !v65 )
        {
          if ( v103 )
            goto LABEL_12;
          v66 = v18 == v17;
          goto LABEL_116;
        }
LABEL_76:
        v17 = sub_AA54C0(v17);
        if ( v18 == v17 || !v17 )
          break;
        if ( v103 )
          goto LABEL_8;
      }
      v66 = v17 != 0 && v18 == v17;
      if ( v103 )
        goto LABEL_98;
      v64 = v100;
LABEL_116:
      _libc_free(v64);
    }
    else
    {
      v66 = v17 != 0 && v17 == v18;
    }
LABEL_98:
    if ( v66 )
      goto LABEL_15;
    if ( (*(_BYTE *)(a1 + 88) & 1) != 0 )
    {
LABEL_13:
      v21 = a1 + 96;
      v22 = 3;
LABEL_14:
      v23 = v22 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v10 = *(_QWORD *)(v21 + 72LL * v23);
      if ( v16 == v10 )
        goto LABEL_15;
      v76 = 1;
      while ( v10 != -4096 )
      {
        a5 = (unsigned int)(v76 + 1);
        v23 = v22 & (v76 + v23);
        v10 = *(_QWORD *)(v21 + 72LL * v23);
        if ( v16 == v10 )
          goto LABEL_15;
        ++v76;
      }
      goto LABEL_107;
    }
LABEL_100:
    v22 = *(unsigned int *)(a1 + 104);
    v21 = *(_QWORD *)(a1 + 96);
    if ( (_DWORD)v22 )
    {
      v22 = (unsigned int)(v22 - 1);
      goto LABEL_14;
    }
LABEL_107:
    if ( !v86 || a4 && !(unsigned __int8)sub_D48480(v86, a4, v22, v10) )
      goto LABEL_85;
    v77 = *(_QWORD *)(v16 + 40);
    if ( *(_BYTE *)(v86 + 84) )
    {
      v78 = *(_QWORD **)(v86 + 64);
      v79 = &v78[*(unsigned int *)(v86 + 76)];
      if ( v78 == v79 )
        goto LABEL_85;
      while ( v77 != *v78 )
      {
        if ( v79 == ++v78 )
        {
          v29 = (int)v96;
          goto LABEL_20;
        }
      }
    }
    else if ( !sub_C8CA60(v86 + 56, v77) )
    {
      goto LABEL_85;
    }
LABEL_15:
    sub_2B5C870((__int64)&v99, (__int64)&v108, (int *)&v96, v10, a5);
    if ( !(_BYTE)v101 )
      goto LABEL_85;
    v24 = (unsigned int)v106;
    v25 = v106;
    if ( (unsigned int)v106 >= (unsigned __int64)HIDWORD(v106) )
    {
      v10 = (unsigned int)v96;
      a5 = (unsigned int)v106 + 1LL;
      v67 = (unsigned int)v96 | v83 & 0xFFFFFFFF00000000LL;
      v83 = v67;
      if ( HIDWORD(v106) < a5 )
      {
        sub_C8D5F0((__int64)&v105, v107, (unsigned int)v106 + 1LL, 0x10u, a5, a6);
        v24 = (unsigned int)v106;
      }
      v68 = &v105[16 * v24];
      *v68 = v16;
      v68[1] = v67;
      LODWORD(v106) = v106 + 1;
LABEL_85:
      v29 = (int)v96;
      goto LABEL_20;
    }
    v26 = 16LL * (unsigned int)v106;
    v27 = &v105[v26] == 0;
    v28 = &v105[v26];
    v29 = (int)v96;
    if ( !v27 )
    {
      *(_QWORD *)v28 = v16;
      v29 = (int)v96;
      *((_DWORD *)v28 + 2) = (_DWORD)v96;
      v25 = v106;
    }
    v10 = (unsigned int)(v25 + 1);
    LODWORD(v106) = v10;
LABEL_20:
    v15 = v29 + 1;
    LODWORD(v96) = v15;
  }
  while ( v15 < (int)a3 );
  v8 = v84;
LABEL_22:
  v93 = a1;
  v87 = a3;
  v30 = (__int64 **)sub_2B08680(v8, a3);
  v34 = (_QWORD *)sub_ACADE0(v30);
  v96 = (int *)v98;
  v97 = 0xC00000000LL;
  v99 = &v101;
  v100 = 0xC00000000LL;
  if ( a3 )
  {
    v35 = &v101;
    v36 = &v101;
    if ( a3 > 0xC )
    {
      sub_C8D5F0((__int64)&v99, &v101, a3, 4u, v32, v33);
      v35 = v99;
      v36 = (__int64 *)((char *)v99 + 4 * (unsigned int)v100);
      v37 = (__int64 *)((char *)v99 + 4 * a3);
      if ( v37 != v36 )
      {
        do
        {
LABEL_25:
          if ( v36 )
            *(_DWORD *)v36 = 0;
          v36 = (__int64 *)((char *)v36 + 4);
        }
        while ( v37 != v36 );
        v35 = v99;
      }
    }
    else
    {
      v37 = (__int64 *)((char *)&v101 + 4 * a3);
      if ( v37 != &v101 )
        goto LABEL_25;
    }
    LODWORD(v100) = a3;
    if ( 4LL * (unsigned int)a3 )
    {
      v38 = 0;
      v31 = (4 * (unsigned __int64)(unsigned int)a3 - 4) >> 2;
      do
      {
        v39 = v38;
        *((_DWORD *)v35 + v38) = v38;
        ++v38;
      }
      while ( v31 != v39 );
    }
    if ( a4 )
    {
      v40 = a4;
      if ( *(_BYTE *)a4 != 92 )
      {
LABEL_34:
        if ( (int)a3 <= 0 )
          goto LABEL_44;
        goto LABEL_35;
      }
LABEL_124:
      v40 = a4;
      if ( **(_BYTE **)(a4 - 32) == 13 )
      {
        v40 = *(_QWORD *)(a4 - 64);
        if ( v30 == *(__int64 ***)(v40 + 8) )
        {
          v81 = *(_BYTE **)(a4 + 72);
          v82 = *(unsigned int *)(a4 + 80);
          LODWORD(v100) = 0;
          sub_2B35330((__int64)&v99, v81, &v81[4 * v82], v31, v32, v33);
        }
        else
        {
          v40 = a4;
        }
      }
      goto LABEL_34;
    }
    if ( (int)a3 <= 0 )
      goto LABEL_55;
    v40 = 0;
LABEL_35:
    v89 = v40;
    v41 = 0;
    while ( 2 )
    {
      if ( v115 )
      {
        v72 = v112;
        if ( v112 )
        {
          v73 = &v111;
          do
          {
            if ( *(_DWORD *)(v72 + 32) < (int)v41 )
            {
              v72 = *(_QWORD *)(v72 + 24);
            }
            else
            {
              v73 = (int *)v72;
              v72 = *(_QWORD *)(v72 + 16);
            }
          }
          while ( v72 );
          if ( v73 != &v111 && v73[8] <= (int)v41 )
            goto LABEL_42;
        }
      }
      else
      {
        v42 = v108;
        v43 = &v108[4 * (unsigned int)v109];
        if ( v108 != (_BYTE *)v43 )
        {
          while ( *v42 != (_DWORD)v41 )
          {
            if ( v43 == ++v42 )
              goto LABEL_88;
          }
          if ( v42 != v43 )
          {
LABEL_42:
            if ( v91 <= (int)++v41 )
            {
              v40 = v89;
              if ( v89 )
                goto LABEL_44;
              goto LABEL_55;
            }
            continue;
          }
        }
      }
      break;
    }
LABEL_88:
    if ( (unsigned __int8)sub_2B0D8B0(*(unsigned __int8 **)(a2 + 8 * v41)) )
    {
      if ( *v71 != 13 )
      {
        v34 = (_QWORD *)sub_2B7DBF0(&v93, v34, (__int64)v71, v41, v8);
        *((_DWORD *)v99 + v41) = v87 + v41;
      }
    }
    else
    {
      v74 = (unsigned int)v97;
      v75 = (unsigned int)v97 + 1LL;
      if ( v75 > HIDWORD(v97) )
      {
        sub_C8D5F0((__int64)&v96, v98, v75, 4u, v69, v70);
        v74 = (unsigned int)v97;
      }
      v96[v74] = v41;
      LODWORD(v97) = v97 + 1;
    }
    goto LABEL_42;
  }
  v40 = a4;
  if ( !a4 )
    goto LABEL_55;
  if ( *(_BYTE *)a4 == 92 )
    goto LABEL_124;
LABEL_44:
  if ( *(_BYTE *)v34 == 13 )
  {
    v34 = (_QWORD *)a4;
    goto LABEL_55;
  }
  v34 = (_QWORD *)a7(a8, v40, v34, v99, (unsigned int)v100);
  if ( *(_BYTE *)a4 > 0x1Cu && (unsigned __int8)sub_BD3610(a4, 0) )
  {
    v44 = *(_QWORD **)a1;
    v45 = 8LL * *(unsigned int *)(a1 + 8);
    v46 = (_QWORD *)(*(_QWORD *)a1 + v45);
    v47 = v45 >> 3;
    v48 = v45 >> 5;
    if ( v48 )
    {
      v49 = &v44[4 * v48];
      while ( a4 != *(_QWORD *)(*v44 + 96LL) )
      {
        if ( a4 == *(_QWORD *)(v44[1] + 96LL) )
        {
          ++v44;
          goto LABEL_54;
        }
        if ( a4 == *(_QWORD *)(v44[2] + 96LL) )
        {
          v44 += 2;
          goto LABEL_54;
        }
        if ( a4 == *(_QWORD *)(v44[3] + 96LL) )
        {
          v44 += 3;
          goto LABEL_54;
        }
        v44 += 4;
        if ( v44 == v49 )
        {
          v47 = v46 - v44;
          goto LABEL_134;
        }
      }
      goto LABEL_54;
    }
LABEL_134:
    switch ( v47 )
    {
      case 2LL:
LABEL_149:
        if ( a4 != *(_QWORD *)(*v44 + 96LL) )
        {
          ++v44;
LABEL_137:
          if ( a4 != *(_QWORD *)(*v44 + 96LL) )
          {
LABEL_138:
            v94 = a4;
            sub_2400480((__int64)v95, a1 + 1976, &v94);
            goto LABEL_55;
          }
        }
        break;
      case 3LL:
        if ( a4 != *(_QWORD *)(*v44 + 96LL) )
        {
          ++v44;
          goto LABEL_149;
        }
        break;
      case 1LL:
        goto LABEL_137;
      default:
        goto LABEL_138;
    }
LABEL_54:
    if ( v46 != v44 )
      goto LABEL_55;
    goto LABEL_138;
  }
LABEL_55:
  v50 = v96;
  v51 = &v96[(unsigned int)v97];
  if ( v51 != v96 )
  {
    v52 = v34;
    do
    {
      v53 = *v50++;
      v54 = sub_2B7DBF0(&v93, v52, *(_QWORD *)(a2 + 8 * v53), v53, v8);
      v52 = (_QWORD *)v54;
    }
    while ( v51 != v50 );
    v34 = (_QWORD *)v54;
  }
  v55 = (unsigned __int64)v105;
  v56 = &v105[16 * (unsigned int)v106];
  if ( v56 != v105 )
  {
    v57 = v34;
    do
    {
      v58 = *(_DWORD *)(v55 + 8);
      v59 = *(_QWORD *)v55;
      v55 += 16LL;
      v60 = sub_2B7DBF0(&v93, v57, v59, v58, v8);
      v57 = (_QWORD *)v60;
    }
    while ( v56 != (_BYTE *)v55 );
    v34 = (_QWORD *)v60;
  }
  if ( v99 != &v101 )
    _libc_free((unsigned __int64)v99);
  if ( v96 != (int *)v98 )
    _libc_free((unsigned __int64)v96);
  v61 = v112;
  while ( v61 )
  {
    sub_2B10340(*(_QWORD *)(v61 + 24));
    v62 = v61;
    v61 = *(_QWORD *)(v61 + 16);
    j_j___libc_free_0(v62);
  }
  if ( v108 != v110 )
    _libc_free((unsigned __int64)v108);
  if ( v105 != v107 )
    _libc_free((unsigned __int64)v105);
  return v34;
}
