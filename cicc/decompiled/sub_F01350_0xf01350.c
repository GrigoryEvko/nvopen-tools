// Function: sub_F01350
// Address: 0xf01350
//
__int64 __fastcall sub_F01350(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v8; // r12
  __int64 v9; // r9
  _QWORD *v10; // rax
  __int64 v11; // r8
  __int64 v12; // rbx
  __int64 v13; // r9
  __int64 v14; // rax
  _QWORD *v15; // rax
  unsigned __int64 v16; // rcx
  __int64 v17; // rax
  _QWORD *v18; // rax
  unsigned __int64 v19; // rcx
  __int64 v20; // rax
  _QWORD *v21; // rax
  unsigned __int64 v22; // rcx
  __int64 v23; // rax
  _QWORD *v24; // rax
  unsigned __int64 v25; // rcx
  __int64 v26; // rax
  _QWORD *v27; // rax
  unsigned int v28; // eax
  __int64 v29; // r9
  volatile signed __int32 *v30; // rdi
  volatile signed __int32 *v31; // r8
  bool v32; // zf
  signed __int32 v33; // eax
  int v34; // eax
  _QWORD *v35; // rax
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rax
  _QWORD *v39; // rax
  unsigned __int64 v40; // rcx
  __int64 v41; // rax
  _QWORD *v42; // rax
  unsigned __int64 v43; // rcx
  __int64 v44; // rax
  _QWORD *v45; // rax
  unsigned __int64 v46; // rcx
  __int64 v47; // rax
  _QWORD *v48; // rax
  unsigned int v49; // eax
  __int64 v50; // r9
  volatile signed __int32 *v51; // rdi
  volatile signed __int32 *v52; // r8
  signed __int32 v53; // eax
  signed __int32 v54; // eax
  _QWORD *v55; // rax
  __int64 v56; // r8
  __int64 v57; // r9
  __int64 v58; // rax
  _QWORD *v59; // rax
  unsigned __int64 v60; // rcx
  __int64 v61; // rax
  _QWORD *v62; // rax
  unsigned int v63; // eax
  __int64 v64; // r9
  volatile signed __int32 *v65; // rdi
  volatile signed __int32 *v66; // r8
  signed __int32 v67; // eax
  signed __int32 v68; // eax
  _QWORD *v69; // rax
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 v72; // rax
  _QWORD *v73; // rax
  unsigned __int64 v74; // rcx
  __int64 v75; // rax
  _QWORD *v76; // rax
  unsigned __int64 v77; // rcx
  __int64 v78; // rax
  _QWORD *v79; // rax
  unsigned __int64 v80; // rcx
  __int64 v81; // rax
  _QWORD *v82; // rax
  unsigned __int64 v83; // rcx
  __int64 v84; // rax
  _QWORD *v85; // rax
  unsigned __int64 v86; // rcx
  __int64 v87; // rax
  _QWORD *v88; // rax
  unsigned int v89; // eax
  __int64 v90; // r9
  volatile signed __int32 *v91; // rdi
  volatile signed __int32 *v92; // r8
  signed __int32 v93; // eax
  signed __int32 v94; // eax
  _QWORD *v95; // rax
  __int64 v96; // r8
  __int64 v97; // r14
  __int64 v98; // r9
  __int64 v99; // rax
  _QWORD *v100; // rax
  unsigned __int64 v101; // rcx
  __int64 v102; // rax
  _QWORD *v103; // rax
  unsigned __int64 v104; // rcx
  __int64 v105; // rax
  _QWORD *v106; // rax
  volatile signed __int32 *v107; // r14
  unsigned int v108; // eax
  volatile signed __int32 *v109; // r12
  unsigned int *v110; // r8
  signed __int32 v111; // eax
  __int64 result; // rax
  signed __int32 v113; // eax
  signed __int32 v114; // eax
  signed __int32 v115; // eax
  signed __int32 v116; // eax
  signed __int32 v117; // eax
  signed __int32 v118; // eax
  signed __int32 v119; // eax
  signed __int32 v120; // eax
  signed __int32 v121; // eax
  unsigned __int64 v122; // rdx
  unsigned __int64 v123; // rdx
  unsigned __int64 v124; // rdx
  unsigned __int64 v125; // rdx
  unsigned __int64 v126; // rdx
  volatile signed __int32 *v127; // [rsp+10h] [rbp-50h]
  volatile signed __int32 *v128; // [rsp+10h] [rbp-50h]
  volatile signed __int32 *v129; // [rsp+10h] [rbp-50h]
  volatile signed __int32 *v130; // [rsp+10h] [rbp-50h]
  __int64 v131; // [rsp+10h] [rbp-50h]
  __int64 v132; // [rsp+10h] [rbp-50h]
  __int64 v133; // [rsp+10h] [rbp-50h]
  __int64 v134; // [rsp+10h] [rbp-50h]
  __int64 v135; // [rsp+10h] [rbp-50h]
  __int64 v136; // [rsp+10h] [rbp-50h]
  __int64 v137; // [rsp+10h] [rbp-50h]
  __int64 v138; // [rsp+10h] [rbp-50h]
  __int64 v139; // [rsp+10h] [rbp-50h]
  volatile signed __int32 *v140; // [rsp+18h] [rbp-48h]
  volatile signed __int32 *v141; // [rsp+18h] [rbp-48h]
  volatile signed __int32 *v142; // [rsp+18h] [rbp-48h]
  volatile signed __int32 *v143; // [rsp+18h] [rbp-48h]
  unsigned int *v144; // [rsp+18h] [rbp-48h]
  volatile signed __int32 *v145; // [rsp+18h] [rbp-48h]
  volatile signed __int32 *v146; // [rsp+18h] [rbp-48h]
  volatile signed __int32 *v147; // [rsp+18h] [rbp-48h]
  volatile signed __int32 *v148; // [rsp+18h] [rbp-48h]
  volatile signed __int32 *v149; // [rsp+18h] [rbp-48h]
  __int64 v150; // [rsp+18h] [rbp-48h]
  __int64 v151; // [rsp+18h] [rbp-48h]
  __int64 v152; // [rsp+18h] [rbp-48h]
  __int64 v153; // [rsp+18h] [rbp-48h]
  __int64 v154; // [rsp+18h] [rbp-48h]
  __int64 v155; // [rsp+18h] [rbp-48h]
  __int64 v156; // [rsp+18h] [rbp-48h]
  __int64 v157; // [rsp+18h] [rbp-48h]
  __int64 v158; // [rsp+18h] [rbp-48h]
  __int64 v159; // [rsp+18h] [rbp-48h]
  __int64 v160; // [rsp+18h] [rbp-48h]
  __int64 v161; // [rsp+18h] [rbp-48h]
  __int64 v162; // [rsp+18h] [rbp-48h]
  __int64 v163; // [rsp+18h] [rbp-48h]
  __int64 v164; // [rsp+18h] [rbp-48h]
  __int64 v165; // [rsp+20h] [rbp-40h] BYREF
  volatile signed __int32 *v166; // [rsp+28h] [rbp-38h]

  v6 = a1 + 1048;
  v8 = a1 + 1576;
  sub_EFD2C0(9u, a1 + 1576, (_DWORD *)(a1 + 1048), (__int64)"Remark", qword_497B3B8, a6);
  sub_EFCCF0(5u, a1 + 1576, a1 + 1048, (__int64)"Remark header", qword_497B368, v9);
  v10 = (_QWORD *)sub_22077B0(544);
  v11 = (__int64)v10;
  if ( v10 )
  {
    v12 = (__int64)(v10 + 2);
    v13 = 5;
    v10[1] = 0x100000001LL;
    *v10 = &unk_49D9900;
    v10[2] = v10 + 4;
    v10[3] = 0x2000000000LL;
    v14 = 0;
  }
  else
  {
    v14 = MEMORY[0x18];
    v12 = 16;
    v13 = 5;
    v123 = MEMORY[0x18] + 1LL;
    if ( v123 > MEMORY[0x1C] )
    {
      sub_C8D5F0(16, (const void *)0x20, v123, 0x10u, 0, 5);
      v14 = MEMORY[0x18];
      v13 = 5;
      v11 = 0;
    }
  }
  v15 = (_QWORD *)(*(_QWORD *)(v11 + 16) + 16 * v14);
  *v15 = 5;
  v15[1] = 1;
  v16 = *(unsigned int *)(v11 + 28);
  v17 = (unsigned int)(*(_DWORD *)(v11 + 24) + 1);
  *(_DWORD *)(v11 + 24) = v17;
  if ( v17 + 1 > v16 )
  {
    v153 = v11;
    sub_C8D5F0(v12, (const void *)(v11 + 32), v17 + 1, 0x10u, v11, 5);
    v11 = v153;
    v17 = *(unsigned int *)(v153 + 24);
  }
  v18 = (_QWORD *)(*(_QWORD *)(v11 + 16) + 16 * v17);
  *v18 = 3;
  v18[1] = 2;
  v19 = *(unsigned int *)(v11 + 28);
  v20 = (unsigned int)(*(_DWORD *)(v11 + 24) + 1);
  *(_DWORD *)(v11 + 24) = v20;
  if ( v20 + 1 > v19 )
  {
    v154 = v11;
    sub_C8D5F0(v12, (const void *)(v11 + 32), v20 + 1, 0x10u, v11, v13);
    v11 = v154;
    v20 = *(unsigned int *)(v154 + 24);
  }
  v21 = (_QWORD *)(*(_QWORD *)(v11 + 16) + 16 * v20);
  *v21 = 6;
  v21[1] = 4;
  v22 = *(unsigned int *)(v11 + 28);
  v23 = (unsigned int)(*(_DWORD *)(v11 + 24) + 1);
  *(_DWORD *)(v11 + 24) = v23;
  if ( v23 + 1 > v22 )
  {
    v155 = v11;
    sub_C8D5F0(v12, (const void *)(v11 + 32), v23 + 1, 0x10u, v11, v13);
    v11 = v155;
    v23 = *(unsigned int *)(v155 + 24);
  }
  v24 = (_QWORD *)(*(_QWORD *)(v11 + 16) + 16 * v23);
  *v24 = 6;
  v24[1] = 4;
  v25 = *(unsigned int *)(v11 + 28);
  v26 = (unsigned int)(*(_DWORD *)(v11 + 24) + 1);
  *(_DWORD *)(v11 + 24) = v26;
  if ( v26 + 1 > v25 )
  {
    v160 = v11;
    sub_C8D5F0(v12, (const void *)(v11 + 32), v26 + 1, 0x10u, v11, v13);
    v11 = v160;
    v26 = *(unsigned int *)(v160 + 24);
  }
  v27 = (_QWORD *)(*(_QWORD *)(v11 + 16) + 16 * v26);
  *v27 = 8;
  v27[1] = 4;
  v165 = v12;
  ++*(_DWORD *)(v11 + 24);
  v166 = (volatile signed __int32 *)v11;
  v140 = (volatile signed __int32 *)(v11 + 8);
  if ( &_pthread_key_create )
    _InterlockedAdd((volatile signed __int32 *)(v11 + 8), 1u);
  else
    ++*(_DWORD *)(v11 + 8);
  v127 = (volatile signed __int32 *)v11;
  v28 = sub_A1A630(v8, 9, &v165);
  v30 = v166;
  v31 = v127;
  v32 = v166 == 0;
  *(_QWORD *)(a1 + 1768) = v28;
  if ( !v32 )
  {
    if ( &_pthread_key_create )
    {
      v33 = _InterlockedExchangeAdd(v30 + 2, 0xFFFFFFFF);
    }
    else
    {
      v33 = *((_DWORD *)v30 + 2);
      *((_DWORD *)v30 + 2) = v33 - 1;
    }
    if ( v33 == 1 )
    {
      (*(void (**)(void))(*(_QWORD *)v30 + 16LL))();
      v31 = v127;
      if ( &_pthread_key_create )
      {
        v113 = _InterlockedExchangeAdd(v30 + 3, 0xFFFFFFFF);
      }
      else
      {
        v113 = *((_DWORD *)v30 + 3);
        *((_DWORD *)v30 + 3) = v113 - 1;
      }
      if ( v113 == 1 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v30 + 24LL))(v30);
        v31 = v127;
      }
    }
  }
  if ( &_pthread_key_create )
  {
    if ( _InterlockedExchangeAdd(v140, 0xFFFFFFFF) != 1 )
      goto LABEL_19;
  }
  else
  {
    v34 = *((_DWORD *)v31 + 2);
    *((_DWORD *)v31 + 2) = v34 - 1;
    if ( v34 != 1 )
      goto LABEL_19;
  }
  v145 = v31;
  (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v31 + 16LL))(v31);
  if ( &_pthread_key_create )
  {
    v114 = _InterlockedExchangeAdd(v145 + 3, 0xFFFFFFFF);
  }
  else
  {
    v114 = *((_DWORD *)v145 + 3);
    *((_DWORD *)v145 + 3) = v114 - 1;
  }
  if ( v114 == 1 )
    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v145 + 24LL))(v145);
LABEL_19:
  sub_EFCCF0(6u, v8, v6, (__int64)"Remark debug location", qword_497B358, v29);
  v35 = (_QWORD *)sub_22077B0(544);
  v36 = (__int64)v35;
  if ( v35 )
  {
    v37 = (__int64)(v35 + 2);
    v35[1] = 0x100000001LL;
    *v35 = &unk_49D9900;
    v35[2] = v35 + 4;
    v35[3] = 0x2000000000LL;
    v38 = 0;
  }
  else
  {
    v38 = MEMORY[0x18];
    v37 = 16;
    v124 = MEMORY[0x18] + 1LL;
    if ( MEMORY[0x1C] < v124 )
    {
      sub_C8D5F0(16, (const void *)0x20, v124, 0x10u, 0, 16);
      v38 = MEMORY[0x18];
      v36 = 0;
      v37 = 16;
    }
  }
  v39 = (_QWORD *)(*(_QWORD *)(v36 + 16) + 16 * v38);
  *v39 = 6;
  v39[1] = 1;
  v40 = *(unsigned int *)(v36 + 28);
  v41 = (unsigned int)(*(_DWORD *)(v36 + 24) + 1);
  *(_DWORD *)(v36 + 24) = v41;
  if ( v41 + 1 > v40 )
  {
    v133 = v36;
    v157 = v37;
    sub_C8D5F0(v37, (const void *)(v36 + 32), v41 + 1, 0x10u, v36, v37);
    v36 = v133;
    v37 = v157;
    v41 = *(unsigned int *)(v133 + 24);
  }
  v42 = (_QWORD *)(*(_QWORD *)(v36 + 16) + 16 * v41);
  *v42 = 7;
  v42[1] = 4;
  v43 = *(unsigned int *)(v36 + 28);
  v44 = (unsigned int)(*(_DWORD *)(v36 + 24) + 1);
  *(_DWORD *)(v36 + 24) = v44;
  if ( v44 + 1 > v43 )
  {
    v132 = v36;
    v156 = v37;
    sub_C8D5F0(v37, (const void *)(v36 + 32), v44 + 1, 0x10u, v36, v37);
    v36 = v132;
    v37 = v156;
    v44 = *(unsigned int *)(v132 + 24);
  }
  v45 = (_QWORD *)(*(_QWORD *)(v36 + 16) + 16 * v44);
  *v45 = 32;
  v45[1] = 2;
  v46 = *(unsigned int *)(v36 + 28);
  v47 = (unsigned int)(*(_DWORD *)(v36 + 24) + 1);
  *(_DWORD *)(v36 + 24) = v47;
  if ( v47 + 1 > v46 )
  {
    v135 = v36;
    v159 = v37;
    sub_C8D5F0(v37, (const void *)(v36 + 32), v47 + 1, 0x10u, v36, v37);
    v36 = v135;
    v37 = v159;
    v47 = *(unsigned int *)(v135 + 24);
  }
  v48 = (_QWORD *)(*(_QWORD *)(v36 + 16) + 16 * v47);
  *v48 = 32;
  v48[1] = 2;
  ++*(_DWORD *)(v36 + 24);
  v165 = v37;
  v166 = (volatile signed __int32 *)v36;
  v141 = (volatile signed __int32 *)(v36 + 8);
  if ( &_pthread_key_create )
    _InterlockedAdd((volatile signed __int32 *)(v36 + 8), 1u);
  else
    ++*(_DWORD *)(v36 + 8);
  v128 = (volatile signed __int32 *)v36;
  v49 = sub_A1A630(v8, 9, &v165);
  v51 = v166;
  v52 = v128;
  v32 = v166 == 0;
  *(_QWORD *)(a1 + 1776) = v49;
  if ( !v32 )
  {
    if ( &_pthread_key_create )
    {
      v53 = _InterlockedExchangeAdd(v51 + 2, 0xFFFFFFFF);
    }
    else
    {
      v53 = *((_DWORD *)v51 + 2);
      *((_DWORD *)v51 + 2) = v53 - 1;
    }
    if ( v53 == 1 )
    {
      (*(void (**)(void))(*(_QWORD *)v51 + 16LL))();
      v52 = v128;
      if ( &_pthread_key_create )
      {
        v121 = _InterlockedExchangeAdd(v51 + 3, 0xFFFFFFFF);
      }
      else
      {
        v121 = *((_DWORD *)v51 + 3);
        *((_DWORD *)v51 + 3) = v121 - 1;
      }
      if ( v121 == 1 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v51 + 24LL))(v51);
        v52 = v128;
      }
    }
  }
  if ( &_pthread_key_create )
  {
    v54 = _InterlockedExchangeAdd(v141, 0xFFFFFFFF);
  }
  else
  {
    v54 = *((_DWORD *)v52 + 2);
    *((_DWORD *)v52 + 2) = v54 - 1;
  }
  if ( v54 == 1 )
  {
    v149 = v52;
    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v52 + 16LL))(v52);
    if ( &_pthread_key_create )
    {
      v118 = _InterlockedExchangeAdd(v149 + 3, 0xFFFFFFFF);
    }
    else
    {
      v118 = *((_DWORD *)v149 + 3);
      *((_DWORD *)v149 + 3) = v118 - 1;
    }
    if ( v118 == 1 )
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v149 + 24LL))(v149);
  }
  sub_EFCCF0(7u, v8, v6, (__int64)"Remark hotness", qword_497B348, v50);
  v55 = (_QWORD *)sub_22077B0(544);
  v56 = (__int64)v55;
  if ( v55 )
  {
    v57 = (__int64)(v55 + 2);
    v55[1] = 0x100000001LL;
    *v55 = &unk_49D9900;
    v55[2] = v55 + 4;
    v55[3] = 0x2000000000LL;
    v58 = 0;
  }
  else
  {
    v58 = MEMORY[0x18];
    v57 = 16;
    v125 = MEMORY[0x18] + 1LL;
    if ( MEMORY[0x1C] < v125 )
    {
      sub_C8D5F0(16, (const void *)0x20, v125, 0x10u, 0, 16);
      v58 = MEMORY[0x18];
      v56 = 0;
      v57 = 16;
    }
  }
  v59 = (_QWORD *)(*(_QWORD *)(v56 + 16) + 16 * v58);
  *v59 = 7;
  v59[1] = 1;
  v60 = *(unsigned int *)(v56 + 28);
  v61 = (unsigned int)(*(_DWORD *)(v56 + 24) + 1);
  *(_DWORD *)(v56 + 24) = v61;
  if ( v61 + 1 > v60 )
  {
    v134 = v56;
    v158 = v57;
    sub_C8D5F0(v57, (const void *)(v56 + 32), v61 + 1, 0x10u, v56, v57);
    v56 = v134;
    v57 = v158;
    v61 = *(unsigned int *)(v134 + 24);
  }
  v62 = (_QWORD *)(*(_QWORD *)(v56 + 16) + 16 * v61);
  *v62 = 8;
  v62[1] = 4;
  ++*(_DWORD *)(v56 + 24);
  v165 = v57;
  v166 = (volatile signed __int32 *)v56;
  v142 = (volatile signed __int32 *)(v56 + 8);
  if ( &_pthread_key_create )
    _InterlockedAdd((volatile signed __int32 *)(v56 + 8), 1u);
  else
    ++*(_DWORD *)(v56 + 8);
  v129 = (volatile signed __int32 *)v56;
  v63 = sub_A1A630(v8, 9, &v165);
  v65 = v166;
  v66 = v129;
  v32 = v166 == 0;
  *(_QWORD *)(a1 + 1784) = v63;
  if ( !v32 )
  {
    if ( &_pthread_key_create )
    {
      v67 = _InterlockedExchangeAdd(v65 + 2, 0xFFFFFFFF);
    }
    else
    {
      v67 = *((_DWORD *)v65 + 2);
      *((_DWORD *)v65 + 2) = v67 - 1;
    }
    if ( v67 == 1 )
    {
      (*(void (**)(void))(*(_QWORD *)v65 + 16LL))();
      v66 = v129;
      if ( &_pthread_key_create )
      {
        v120 = _InterlockedExchangeAdd(v65 + 3, 0xFFFFFFFF);
      }
      else
      {
        v120 = *((_DWORD *)v65 + 3);
        *((_DWORD *)v65 + 3) = v120 - 1;
      }
      if ( v120 == 1 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v65 + 24LL))(v65);
        v66 = v129;
      }
    }
  }
  if ( &_pthread_key_create )
  {
    v68 = _InterlockedExchangeAdd(v142, 0xFFFFFFFF);
  }
  else
  {
    v68 = *((_DWORD *)v66 + 2);
    *((_DWORD *)v66 + 2) = v68 - 1;
  }
  if ( v68 == 1 )
  {
    v148 = v66;
    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v66 + 16LL))(v66);
    if ( &_pthread_key_create )
    {
      v117 = _InterlockedExchangeAdd(v148 + 3, 0xFFFFFFFF);
    }
    else
    {
      v117 = *((_DWORD *)v148 + 3);
      *((_DWORD *)v148 + 3) = v117 - 1;
    }
    if ( v117 == 1 )
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v148 + 24LL))(v148);
  }
  sub_EFCCF0(8u, v8, v6, (__int64)"Argument with debug location", qword_497B338, v64);
  v69 = (_QWORD *)sub_22077B0(544);
  v70 = (__int64)v69;
  if ( v69 )
  {
    v71 = (__int64)(v69 + 2);
    v69[1] = 0x100000001LL;
    *v69 = &unk_49D9900;
    v69[2] = v69 + 4;
    v69[3] = 0x2000000000LL;
    v72 = 0;
  }
  else
  {
    v72 = MEMORY[0x18];
    v71 = 16;
    v126 = MEMORY[0x18] + 1LL;
    if ( v126 > MEMORY[0x1C] )
    {
      sub_C8D5F0(16, (const void *)0x20, v126, 0x10u, 0, 16);
      v72 = MEMORY[0x18];
      v70 = 0;
      v71 = 16;
    }
  }
  v73 = (_QWORD *)(*(_QWORD *)(v70 + 16) + 16 * v72);
  *v73 = 8;
  v73[1] = 1;
  v74 = *(unsigned int *)(v70 + 28);
  v75 = (unsigned int)(*(_DWORD *)(v70 + 24) + 1);
  *(_DWORD *)(v70 + 24) = v75;
  if ( v75 + 1 > v74 )
  {
    v139 = v70;
    v164 = v71;
    sub_C8D5F0(v71, (const void *)(v70 + 32), v75 + 1, 0x10u, v70, v71);
    v70 = v139;
    v71 = v164;
    v75 = *(unsigned int *)(v139 + 24);
  }
  v76 = (_QWORD *)(*(_QWORD *)(v70 + 16) + 16 * v75);
  *v76 = 7;
  v76[1] = 4;
  v77 = *(unsigned int *)(v70 + 28);
  v78 = (unsigned int)(*(_DWORD *)(v70 + 24) + 1);
  *(_DWORD *)(v70 + 24) = v78;
  if ( v78 + 1 > v77 )
  {
    v138 = v70;
    v163 = v71;
    sub_C8D5F0(v71, (const void *)(v70 + 32), v78 + 1, 0x10u, v70, v71);
    v70 = v138;
    v71 = v163;
    v78 = *(unsigned int *)(v138 + 24);
  }
  v79 = (_QWORD *)(*(_QWORD *)(v70 + 16) + 16 * v78);
  *v79 = 7;
  v79[1] = 4;
  v80 = *(unsigned int *)(v70 + 28);
  v81 = (unsigned int)(*(_DWORD *)(v70 + 24) + 1);
  *(_DWORD *)(v70 + 24) = v81;
  if ( v81 + 1 > v80 )
  {
    v137 = v70;
    v162 = v71;
    sub_C8D5F0(v71, (const void *)(v70 + 32), v81 + 1, 0x10u, v70, v71);
    v70 = v137;
    v71 = v162;
    v81 = *(unsigned int *)(v137 + 24);
  }
  v82 = (_QWORD *)(*(_QWORD *)(v70 + 16) + 16 * v81);
  *v82 = 7;
  v82[1] = 4;
  v83 = *(unsigned int *)(v70 + 28);
  v84 = (unsigned int)(*(_DWORD *)(v70 + 24) + 1);
  *(_DWORD *)(v70 + 24) = v84;
  if ( v84 + 1 > v83 )
  {
    v136 = v70;
    v161 = v71;
    sub_C8D5F0(v71, (const void *)(v70 + 32), v84 + 1, 0x10u, v70, v71);
    v70 = v136;
    v71 = v161;
    v84 = *(unsigned int *)(v136 + 24);
  }
  v85 = (_QWORD *)(*(_QWORD *)(v70 + 16) + 16 * v84);
  *v85 = 32;
  v85[1] = 2;
  v86 = *(unsigned int *)(v70 + 28);
  v87 = (unsigned int)(*(_DWORD *)(v70 + 24) + 1);
  *(_DWORD *)(v70 + 24) = v87;
  if ( v87 + 1 > v86 )
  {
    v131 = v70;
    v150 = v71;
    sub_C8D5F0(v71, (const void *)(v70 + 32), v87 + 1, 0x10u, v70, v71);
    v70 = v131;
    v71 = v150;
    v87 = *(unsigned int *)(v131 + 24);
  }
  v88 = (_QWORD *)(*(_QWORD *)(v70 + 16) + 16 * v87);
  *v88 = 32;
  v88[1] = 2;
  ++*(_DWORD *)(v70 + 24);
  v165 = v71;
  v166 = (volatile signed __int32 *)v70;
  v143 = (volatile signed __int32 *)(v70 + 8);
  if ( &_pthread_key_create )
    _InterlockedAdd((volatile signed __int32 *)(v70 + 8), 1u);
  else
    ++*(_DWORD *)(v70 + 8);
  v130 = (volatile signed __int32 *)v70;
  v89 = sub_A1A630(v8, 9, &v165);
  v91 = v166;
  v92 = v130;
  v32 = v166 == 0;
  *(_QWORD *)(a1 + 1792) = v89;
  if ( !v32 )
  {
    if ( &_pthread_key_create )
    {
      v93 = _InterlockedExchangeAdd(v91 + 2, 0xFFFFFFFF);
    }
    else
    {
      v93 = *((_DWORD *)v91 + 2);
      *((_DWORD *)v91 + 2) = v93 - 1;
    }
    if ( v93 == 1 )
    {
      (*(void (**)(void))(*(_QWORD *)v91 + 16LL))();
      v92 = v130;
      if ( &_pthread_key_create )
      {
        v116 = _InterlockedExchangeAdd(v91 + 3, 0xFFFFFFFF);
      }
      else
      {
        v116 = *((_DWORD *)v91 + 3);
        *((_DWORD *)v91 + 3) = v116 - 1;
      }
      if ( v116 == 1 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v91 + 24LL))(v91);
        v92 = v130;
      }
    }
  }
  if ( &_pthread_key_create )
  {
    v94 = _InterlockedExchangeAdd(v143, 0xFFFFFFFF);
  }
  else
  {
    v94 = *((_DWORD *)v92 + 2);
    *((_DWORD *)v92 + 2) = v94 - 1;
  }
  if ( v94 == 1 )
  {
    v146 = v92;
    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v92 + 16LL))(v92);
    if ( &_pthread_key_create )
    {
      v115 = _InterlockedExchangeAdd(v146 + 3, 0xFFFFFFFF);
    }
    else
    {
      v115 = *((_DWORD *)v146 + 3);
      *((_DWORD *)v146 + 3) = v115 - 1;
    }
    if ( v115 == 1 )
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v146 + 24LL))(v146);
  }
  sub_EFCCF0(9u, v8, v6, (__int64)"Argument", qword_497B328, v90);
  v95 = (_QWORD *)sub_22077B0(544);
  v96 = (__int64)v95;
  if ( v95 )
  {
    v97 = (__int64)(v95 + 2);
    v98 = 1;
    v95[1] = 0x100000001LL;
    *v95 = &unk_49D9900;
    v95[2] = v95 + 4;
    v95[3] = 0x2000000000LL;
    v99 = 0;
  }
  else
  {
    v99 = MEMORY[0x18];
    v97 = 16;
    v98 = 1;
    v122 = MEMORY[0x18] + 1LL;
    if ( v122 > MEMORY[0x1C] )
    {
      sub_C8D5F0(16, (const void *)0x20, v122, 0x10u, 0, 1);
      v99 = MEMORY[0x18];
      v98 = 1;
      v96 = 0;
    }
  }
  v100 = (_QWORD *)(*(_QWORD *)(v96 + 16) + 16 * v99);
  *v100 = 9;
  v100[1] = 1;
  v101 = *(unsigned int *)(v96 + 28);
  v102 = (unsigned int)(*(_DWORD *)(v96 + 24) + 1);
  *(_DWORD *)(v96 + 24) = v102;
  if ( v102 + 1 > v101 )
  {
    v152 = v96;
    sub_C8D5F0(v97, (const void *)(v96 + 32), v102 + 1, 0x10u, v96, 1);
    v96 = v152;
    v102 = *(unsigned int *)(v152 + 24);
  }
  v103 = (_QWORD *)(*(_QWORD *)(v96 + 16) + 16 * v102);
  *v103 = 7;
  v103[1] = 4;
  v104 = *(unsigned int *)(v96 + 28);
  v105 = (unsigned int)(*(_DWORD *)(v96 + 24) + 1);
  *(_DWORD *)(v96 + 24) = v105;
  if ( v105 + 1 > v104 )
  {
    v151 = v96;
    sub_C8D5F0(v97, (const void *)(v96 + 32), v105 + 1, 0x10u, v96, v98);
    v96 = v151;
    v105 = *(unsigned int *)(v151 + 24);
  }
  v106 = (_QWORD *)(*(_QWORD *)(v96 + 16) + 16 * v105);
  *v106 = 7;
  v106[1] = 4;
  v165 = v97;
  v107 = (volatile signed __int32 *)(v96 + 8);
  ++*(_DWORD *)(v96 + 24);
  v166 = (volatile signed __int32 *)v96;
  if ( &_pthread_key_create )
    _InterlockedAdd(v107, 1u);
  else
    ++*(_DWORD *)(v96 + 8);
  v144 = (unsigned int *)v96;
  v108 = sub_A1A630(v8, 9, &v165);
  v109 = v166;
  v110 = v144;
  v32 = v166 == 0;
  *(_QWORD *)(a1 + 1800) = v108;
  if ( !v32 )
  {
    if ( &_pthread_key_create )
    {
      v111 = _InterlockedExchangeAdd(v109 + 2, 0xFFFFFFFF);
    }
    else
    {
      v111 = *((_DWORD *)v109 + 2);
      *((_DWORD *)v109 + 2) = v111 - 1;
    }
    if ( v111 == 1 )
    {
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v109 + 16LL))(v109);
      v110 = v144;
      if ( &_pthread_key_create )
      {
        v119 = _InterlockedExchangeAdd(v109 + 3, 0xFFFFFFFF);
      }
      else
      {
        v119 = *((_DWORD *)v109 + 3);
        *((_DWORD *)v109 + 3) = v119 - 1;
      }
      if ( v119 == 1 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v109 + 24LL))(v109);
        v110 = v144;
      }
    }
  }
  if ( &_pthread_key_create )
  {
    result = (unsigned int)_InterlockedExchangeAdd(v107, 0xFFFFFFFF);
  }
  else
  {
    result = v110[2];
    v110[2] = result - 1;
  }
  if ( (_DWORD)result == 1 )
  {
    v147 = (volatile signed __int32 *)v110;
    (*(void (__fastcall **)(unsigned int *))(*(_QWORD *)v110 + 16LL))(v110);
    if ( &_pthread_key_create )
    {
      result = (unsigned int)_InterlockedExchangeAdd(v147 + 3, 0xFFFFFFFF);
    }
    else
    {
      result = *((unsigned int *)v147 + 3);
      *((_DWORD *)v147 + 3) = result - 1;
    }
    if ( (_DWORD)result == 1 )
      return (*(__int64 (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v147 + 24LL))(v147);
  }
  return result;
}
