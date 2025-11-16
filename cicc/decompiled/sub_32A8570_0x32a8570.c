// Function: sub_32A8570
// Address: 0x32a8570
//
__int64 __fastcall sub_32A8570(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r12
  unsigned __int16 *v9; // rdx
  __int128 *v11; // r15
  int v12; // eax
  __int128 *v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rsi
  int v16; // eax
  __int64 v17; // rdx
  int v18; // ecx
  __int64 *v19; // rax
  int v20; // r9d
  __int64 *v21; // rax
  __int64 v22; // r15
  __int64 v23; // r10
  __int64 v24; // rax
  int v25; // edx
  __int64 v26; // rdx
  __int64 v27; // rax
  int v28; // edx
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // r15
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // rax
  __int128 v37; // rdi
  __int64 v38; // r8
  __int64 v39; // rcx
  __int128 v40; // rax
  __int64 v41; // rax
  int v42; // eax
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rcx
  int v46; // esi
  __int64 *v47; // rdx
  __int64 v48; // rdx
  int v49; // eax
  __int64 *v50; // rax
  __int64 v51; // rax
  int v52; // eax
  __int64 *v53; // rax
  __int64 v54; // r9
  int v55; // r10d
  __int64 v56; // rdx
  int v57; // eax
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 v60; // rcx
  int v61; // esi
  __int64 *v62; // rdx
  __int64 v63; // rdx
  int v64; // eax
  __int64 *v65; // rax
  __int64 v66; // rdi
  unsigned int v67; // r12d
  unsigned __int64 v68; // rdi
  bool v69; // bl
  int v70; // r9d
  __int64 v71; // r13
  unsigned int v72; // edx
  unsigned int v73; // ebx
  int v74; // r9d
  int v75; // edx
  __int128 v76; // rax
  int v77; // r9d
  int v78; // edx
  int v79; // r9d
  __int64 v80; // rax
  __int64 v81; // rdx
  __int64 *v82; // rax
  __int64 v83; // r8
  __int64 v84; // rdx
  __int64 v85; // rsi
  __int128 *v86; // r9
  __int128 v87; // rax
  int v88; // r9d
  __int64 v89; // rax
  __int64 v90; // rax
  __int64 v91; // rdx
  __int128 v92; // [rsp-20h] [rbp-180h]
  __int128 v93; // [rsp-10h] [rbp-170h]
  __int64 v94; // [rsp+8h] [rbp-158h]
  __int64 v95; // [rsp+10h] [rbp-150h]
  __int64 v96; // [rsp+18h] [rbp-148h]
  __int64 v97; // [rsp+20h] [rbp-140h]
  __int64 v99; // [rsp+30h] [rbp-130h]
  int v100; // [rsp+38h] [rbp-128h]
  int v101; // [rsp+3Ch] [rbp-124h]
  __int128 v102; // [rsp+40h] [rbp-120h]
  unsigned __int64 v103; // [rsp+40h] [rbp-120h]
  unsigned int v104; // [rsp+70h] [rbp-F0h] BYREF
  __int128 *v105; // [rsp+78h] [rbp-E8h]
  __int64 v106; // [rsp+80h] [rbp-E0h] BYREF
  int v107; // [rsp+88h] [rbp-D8h]
  __int128 v108; // [rsp+90h] [rbp-D0h] BYREF
  __int128 v109; // [rsp+A0h] [rbp-C0h] BYREF
  __int128 v110; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v111; // [rsp+C0h] [rbp-A0h]
  __int64 v112; // [rsp+C8h] [rbp-98h]
  int v113; // [rsp+D0h] [rbp-90h] BYREF
  __int128 *v114; // [rsp+D8h] [rbp-88h]
  char v115; // [rsp+E8h] [rbp-78h]
  int v116; // [rsp+F0h] [rbp-70h] BYREF
  __int128 *v117; // [rsp+F8h] [rbp-68h]
  __int128 *v118; // [rsp+100h] [rbp-60h]
  char v119; // [rsp+108h] [rbp-58h]
  char v120; // [rsp+10Ch] [rbp-54h]
  unsigned __int64 v121; // [rsp+110h] [rbp-50h]
  unsigned int v122; // [rsp+118h] [rbp-48h]
  char v123; // [rsp+124h] [rbp-3Ch]

  v8 = a2;
  *((_QWORD *)&v102 + 1) = a5;
  *(_QWORD *)&v102 = a4;
  v101 = a5;
  v9 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * a3);
  v11 = (__int128 *)*((_QWORD *)v9 + 1);
  v12 = *v9;
  v105 = v11;
  LOWORD(v104) = v12;
  if ( (_WORD)v12 )
  {
    if ( (unsigned __int16)(v12 - 17) > 0xD3u )
    {
      LOWORD(v116) = v12;
      v117 = v11;
      goto LABEL_4;
    }
    LOWORD(v12) = word_4456580[v12 - 1];
    v13 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v104) )
    {
      v117 = v11;
      LOWORD(v116) = 0;
      goto LABEL_9;
    }
    LOWORD(v12) = sub_3009970((__int64)&v104, a2, v33, v34, v35);
  }
  LOWORD(v116) = v12;
  v117 = v13;
  if ( !(_WORD)v12 )
  {
LABEL_9:
    v111 = sub_3007260((__int64)&v116);
    v112 = v14;
    LODWORD(v96) = v111;
    goto LABEL_10;
  }
LABEL_4:
  if ( (_WORD)v12 == 1 || (unsigned __int16)(v12 - 504) <= 7u )
    BUG();
  v96 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v12 - 16];
LABEL_10:
  v15 = *(_QWORD *)(a6 + 80);
  v106 = v15;
  if ( v15 )
    sub_B96E90((__int64)&v106, v15, 1);
  v107 = *(_DWORD *)(a6 + 72);
  v16 = *(_DWORD *)(v8 + 24);
  if ( ((v16 - 214) & 0xFFFFFFFD) != 0 )
  {
    v18 = *(_DWORD *)(v8 + 24);
    v17 = v8;
  }
  else
  {
    v17 = **(_QWORD **)(v8 + 40);
    v18 = *(_DWORD *)(v17 + 24);
  }
  if ( v18 == 186 )
  {
    if ( ((*(_DWORD *)(a4 + 24) - 214) & 0xFFFFFFFD) != 0 )
    {
      v99 = a4;
      v20 = v101;
    }
    else
    {
      v19 = *(__int64 **)(a4 + 40);
      v20 = *((_DWORD *)v19 + 2);
      v99 = *v19;
    }
    v21 = *(__int64 **)(v17 + 40);
    v22 = v21[5];
    v23 = *v21;
    v97 = v21[6];
    if ( *((_DWORD *)v21 + 2) == v20 && v99 == v23 || *((_DWORD *)v21 + 12) == v20 && v99 == v22 )
    {
      v8 = v102;
      goto LABEL_34;
    }
    v100 = v20;
    v94 = *v21;
    v95 = v21[1];
    v24 = sub_33DFEB0(v22, v97, v23, v95, 0);
    if ( !v24 )
      goto LABEL_26;
    if ( ((*(_DWORD *)(v24 + 24) - 214) & 0xFFFFFFFD) == 0 )
    {
      v26 = *(_QWORD *)(v24 + 40);
      v24 = *(_QWORD *)v26;
      v25 = *(_DWORD *)(v26 + 8);
    }
    if ( v100 == v25 && v99 == v24 )
    {
      v83 = v104;
      v84 = v95;
      v85 = v94;
      v86 = v105;
    }
    else
    {
LABEL_26:
      v27 = sub_33DFEB0(v94, v95, v22, v97, 0);
      if ( !v27 )
        goto LABEL_31;
      if ( ((*(_DWORD *)(v27 + 24) - 214) & 0xFFFFFFFD) == 0 )
      {
        v29 = *(_QWORD *)(v27 + 40);
        v27 = *(_QWORD *)v29;
        v28 = *(_DWORD *)(v29 + 8);
      }
      if ( v100 != v28 || v99 != v27 )
      {
LABEL_31:
        if ( *(_DWORD *)(v8 + 24) != 188 )
          goto LABEL_32;
        v51 = *(_QWORD *)(v8 + 40);
        *(_QWORD *)&v37 = *(_QWORD *)v51;
        *((_QWORD *)&v37 + 1) = *(unsigned int *)(v51 + 8);
        v38 = *(_QWORD *)(v51 + 40);
        v39 = *(unsigned int *)(v51 + 48);
        goto LABEL_62;
      }
      v84 = v97;
      v85 = v22;
      v83 = v104;
      v86 = v105;
    }
    *(_QWORD *)&v87 = sub_33FB310(a1, v85, v84, &v106, v83, v86);
    v41 = sub_3406EB0(a1, 187, (unsigned int)&v106, v104, (_DWORD)v105, v88, v87, v102);
    goto LABEL_45;
  }
  if ( v16 != 188 )
    goto LABEL_32;
  v36 = *(_QWORD *)(v8 + 40);
  *(_QWORD *)&v37 = *(_QWORD *)v36;
  *((_QWORD *)&v37 + 1) = *(unsigned int *)(v36 + 8);
  v38 = *(_QWORD *)(v36 + 40);
  v39 = *(unsigned int *)(v36 + 48);
  if ( !a4 )
  {
    if ( !v38 )
    {
      if ( !(_QWORD)v37 )
        goto LABEL_65;
LABEL_43:
      *(_QWORD *)&v40 = v38;
      v93 = v102;
      *((_QWORD *)&v40 + 1) = v39;
LABEL_44:
      v41 = sub_3406EB0(a1, 187, (unsigned int)&v106, v104, (_DWORD)v105, (unsigned int)&v106, v40, v93);
LABEL_45:
      v8 = v41;
      goto LABEL_34;
    }
LABEL_60:
    v39 = DWORD2(v37);
    v38 = v37;
    goto LABEL_43;
  }
LABEL_62:
  if ( (_DWORD)v39 == v101 && a4 == v38 )
    goto LABEL_60;
  if ( a4 == (_QWORD)v37 && DWORD2(v37) == v101 )
    goto LABEL_43;
LABEL_65:
  v52 = *(_DWORD *)(a4 + 24);
  if ( v52 != 186 )
  {
    if ( v52 != 187 )
      goto LABEL_32;
    v53 = *(__int64 **)(a4 + 40);
    v54 = *v53;
    v55 = *((_DWORD *)v53 + 2);
    v56 = v53[5];
    if ( (_QWORD)v37 )
    {
      v57 = *((_DWORD *)v53 + 12);
      if ( DWORD2(v37) != v55 || v54 != (_QWORD)v37 )
        goto LABEL_70;
LABEL_117:
      if ( v38 )
      {
        if ( (_DWORD)v39 == v57 && v56 == v38 )
          goto LABEL_99;
        if ( (_QWORD)v37 )
        {
LABEL_70:
          if ( v56 != (_QWORD)v37 )
            goto LABEL_32;
          goto LABEL_96;
        }
LABEL_111:
        if ( !v56 )
          goto LABEL_32;
        goto LABEL_112;
      }
LABEL_131:
      if ( v56 )
        goto LABEL_99;
      goto LABEL_32;
    }
    if ( v54 )
    {
      v57 = *((_DWORD *)v53 + 12);
      goto LABEL_117;
    }
LABEL_128:
    if ( !v56 || !v38 )
      goto LABEL_32;
    goto LABEL_112;
  }
  v82 = *(__int64 **)(a4 + 40);
  v54 = *v82;
  v55 = *((_DWORD *)v82 + 2);
  v56 = v82[5];
  if ( (_QWORD)v37 )
  {
    v57 = *((_DWORD *)v82 + 12);
    if ( v55 != DWORD2(v37) || v54 != (_QWORD)v37 )
      goto LABEL_95;
  }
  else
  {
    if ( !v54 )
      goto LABEL_128;
    v57 = *((_DWORD *)v82 + 12);
  }
  if ( !v38 )
    goto LABEL_131;
  if ( (_DWORD)v39 == v57 && v38 == v56 )
    goto LABEL_99;
  if ( !(_QWORD)v37 )
    goto LABEL_111;
LABEL_95:
  if ( (_QWORD)v37 != v56 )
    goto LABEL_32;
LABEL_96:
  if ( DWORD2(v37) != v57 )
    goto LABEL_32;
  if ( v38 )
  {
LABEL_112:
    if ( v38 == v54 && (_DWORD)v39 == v55 )
      goto LABEL_99;
    goto LABEL_32;
  }
  if ( v54 )
  {
LABEL_99:
    v40 = v37;
    *((_QWORD *)&v93 + 1) = v39;
    *(_QWORD *)&v93 = v38;
    goto LABEL_44;
  }
LABEL_32:
  v30 = sub_32772A0(a6, v8, a3, v102, SDWORD2(v102), a1);
  v31 = v30;
  if ( v30 )
  {
LABEL_33:
    v8 = v30;
    goto LABEL_34;
  }
  v42 = *(_DWORD *)(v8 + 24);
  if ( v42 != 195 )
  {
    if ( v42 == 196 && *(_DWORD *)(a4 + 24) == 192 )
    {
      v43 = *(_QWORD *)(a4 + 40);
      v44 = *(_QWORD *)(v8 + 40);
      if ( *(_QWORD *)(v44 + 40) == *(_QWORD *)v43 && *(_DWORD *)(v44 + 48) == *(_DWORD *)(v43 + 8) )
      {
        v45 = *(_QWORD *)(v43 + 40);
        v46 = *(_DWORD *)(v43 + 48);
        if ( *(_DWORD *)(v45 + 24) == 214 )
        {
          v47 = *(__int64 **)(v45 + 40);
          v45 = *v47;
          v46 = *((_DWORD *)v47 + 2);
        }
        v48 = *(_QWORD *)(v44 + 80);
        v49 = *(_DWORD *)(v44 + 88);
        if ( *(_DWORD *)(v48 + 24) == 214 )
        {
          v50 = *(__int64 **)(v48 + 40);
          v48 = *v50;
          v49 = *((_DWORD *)v50 + 2);
        }
        if ( v46 == v49 && v45 == v48 )
          goto LABEL_34;
      }
    }
LABEL_81:
    v66 = v8;
    v120 = 0;
    v123 = 0;
    v67 = 0;
    v116 = 190;
    *(_QWORD *)&v108 = 0;
    v103 = (unsigned int)v96 >> 1;
    v121 = v103;
    DWORD2(v108) = 0;
    LODWORD(v117) = 215;
    v118 = &v108;
    v122 = 64;
    if ( (unsigned __int8)sub_32A8340(v66, a3, 0, (__int64)&v116)
      && *(_DWORD *)(a4 + 24) == 214
      && (v89 = *(_QWORD *)(a4 + 40),
          v31 = *(_QWORD *)v89,
          v67 = *(_DWORD *)(v89 + 8),
          v103 == sub_3263630(*(_QWORD *)v89, v67))
      && (v90 = *(_QWORD *)(v108 + 48) + 16LL * DWORD2(v108),
          v91 = *(_QWORD *)(v31 + 48) + 16LL * v67,
          *(_WORD *)v91 == *(_WORD *)v90) )
    {
      v69 = *(_WORD *)v91 != 0 || *(_QWORD *)(v91 + 8) == *(_QWORD *)(v90 + 8);
      if ( v122 <= 0x40 || (v68 = v121) == 0 )
      {
LABEL_85:
        if ( v69 )
        {
          v115 = 0;
          *(_QWORD *)&v109 = 0;
          DWORD2(v109) = 0;
          *(_QWORD *)&v110 = 0;
          DWORD2(v110) = 0;
          v113 = 188;
          v114 = &v109;
          if ( (unsigned __int8)sub_32A8450(v31, v67, 0, (__int64)&v113) )
          {
            v119 = 0;
            v116 = 188;
            v117 = &v110;
            if ( (unsigned __int8)sub_32A8450(v108, SDWORD2(v108), 0, (__int64)&v116) )
            {
              v71 = sub_33FAF80(a1, 214, (unsigned int)&v106, v104, (_DWORD)v105, v70, v109);
              v73 = v72;
              *(_QWORD *)&v108 = sub_33FAF80(a1, 215, (unsigned int)&v106, v104, (_DWORD)v105, v74, v110);
              DWORD2(v108) = v75;
              *(_QWORD *)&v76 = sub_3400E40(a1, v103, v104, v105, &v106);
              *(_QWORD *)&v108 = sub_3406EB0(a1, 190, (unsigned int)&v106, v104, (_DWORD)v105, v77, v108, v76);
              DWORD2(v108) = v78;
              *((_QWORD *)&v92 + 1) = v73;
              *(_QWORD *)&v92 = v71;
              v80 = sub_3406EB0(a1, 187, (unsigned int)&v106, v104, (_DWORD)v105, v79, v92, v108);
              v30 = sub_34074A0(a1, &v106, v80, v81, v104, v105);
              goto LABEL_33;
            }
          }
        }
        goto LABEL_89;
      }
    }
    else if ( v122 <= 0x40 || (v68 = v121, v69 = 0, !v121) )
    {
LABEL_89:
      v8 = 0;
      goto LABEL_34;
    }
    j_j___libc_free_0_0(v68);
    goto LABEL_85;
  }
  if ( *(_DWORD *)(a4 + 24) != 190 )
    goto LABEL_81;
  v58 = *(_QWORD *)(a4 + 40);
  v59 = *(_QWORD *)(v8 + 40);
  if ( *(_QWORD *)v59 != *(_QWORD *)v58 || *(_DWORD *)(v59 + 8) != *(_DWORD *)(v58 + 8) )
    goto LABEL_81;
  v60 = *(_QWORD *)(v58 + 40);
  v61 = *(_DWORD *)(v58 + 48);
  if ( *(_DWORD *)(v60 + 24) == 214 )
  {
    v62 = *(__int64 **)(v60 + 40);
    v60 = *v62;
    v61 = *((_DWORD *)v62 + 2);
  }
  v63 = *(_QWORD *)(v59 + 80);
  v64 = *(_DWORD *)(v59 + 88);
  if ( *(_DWORD *)(v63 + 24) == 214 )
  {
    v65 = *(__int64 **)(v63 + 40);
    v63 = *v65;
    v64 = *((_DWORD *)v65 + 2);
  }
  if ( v60 != v63 || v61 != v64 )
    goto LABEL_81;
LABEL_34:
  if ( v106 )
    sub_B91220((__int64)&v106, v106);
  return v8;
}
