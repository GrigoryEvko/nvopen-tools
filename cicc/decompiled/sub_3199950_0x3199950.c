// Function: sub_3199950
// Address: 0x3199950
//
__int64 __fastcall sub_3199950(_BYTE *a1)
{
  __int64 v1; // r12
  __int64 v2; // rax
  __int64 v3; // r15
  unsigned __int8 *v4; // rax
  unsigned __int8 *v5; // r14
  unsigned int *v6; // r15
  unsigned int *v7; // rbx
  __int64 v8; // rdx
  unsigned int v9; // esi
  unsigned __int8 *v10; // rax
  unsigned __int8 *v11; // r13
  unsigned int *v12; // r15
  unsigned int *v13; // rbx
  __int64 v14; // rdx
  unsigned int v15; // esi
  __int64 v16; // rax
  unsigned __int8 *v17; // r15
  __int64 (__fastcall *v18)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  _BYTE *v19; // rbx
  __int64 v20; // r13
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v25; // rdx
  unsigned int *v26; // r13
  unsigned int *v27; // r12
  __int64 v28; // rdx
  unsigned int v29; // esi
  __int64 v30; // r14
  unsigned int v31; // r13d
  __int64 v32; // rax
  unsigned __int8 *v33; // r15
  unsigned int *v34; // rbx
  unsigned int *v35; // r14
  __int64 v36; // rdx
  unsigned int v37; // esi
  unsigned __int8 *v38; // rax
  unsigned __int8 *v39; // r14
  __int64 v40; // rbx
  unsigned int *v41; // rbx
  unsigned int *v42; // r12
  __int64 v43; // rdx
  unsigned int v44; // esi
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 (__fastcall *v47)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 (__fastcall *v48)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  _BYTE *v49; // r15
  unsigned __int8 *v50; // rbx
  __int64 (__fastcall *v51)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  unsigned __int8 *v52; // r14
  __int64 (__fastcall *v53)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  _BYTE *v54; // rbx
  __int64 v55; // r15
  __int64 v56; // r14
  __int64 v57; // rax
  __int64 v58; // rcx
  __int64 v59; // rdx
  unsigned int *v60; // r12
  unsigned int *v61; // r15
  __int64 v62; // rdx
  unsigned int v63; // esi
  unsigned int *v64; // r14
  unsigned int *v65; // rbx
  __int64 v66; // rdx
  unsigned int v67; // esi
  unsigned int *v68; // r12
  unsigned int *v69; // r15
  __int64 v70; // rdx
  unsigned int v71; // esi
  unsigned int *v72; // r15
  unsigned int *v73; // rbx
  __int64 v74; // rdx
  unsigned int v75; // esi
  __int64 v76; // [rsp+0h] [rbp-170h]
  _BYTE *v77; // [rsp+8h] [rbp-168h]
  unsigned __int8 *v78; // [rsp+8h] [rbp-168h]
  _BYTE *v79; // [rsp+10h] [rbp-160h]
  unsigned __int8 *v80; // [rsp+10h] [rbp-160h]
  __int64 v81; // [rsp+10h] [rbp-160h]
  __int64 v82; // [rsp+30h] [rbp-140h]
  __int64 v83; // [rsp+30h] [rbp-140h]
  unsigned __int8 *v84; // [rsp+30h] [rbp-140h]
  __int64 v85; // [rsp+40h] [rbp-130h]
  _BYTE v86[32]; // [rsp+50h] [rbp-120h] BYREF
  __int16 v87; // [rsp+70h] [rbp-100h]
  _BYTE v88[32]; // [rsp+80h] [rbp-F0h] BYREF
  __int16 v89; // [rsp+A0h] [rbp-D0h]
  unsigned int *v90; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v91; // [rsp+B8h] [rbp-B8h]
  _BYTE v92[32]; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v93; // [rsp+E0h] [rbp-90h]
  __int64 v94; // [rsp+E8h] [rbp-88h]
  __int64 v95; // [rsp+F0h] [rbp-80h]
  _QWORD *v96; // [rsp+F8h] [rbp-78h]
  void **v97; // [rsp+100h] [rbp-70h]
  void **v98; // [rsp+108h] [rbp-68h]
  __int64 v99; // [rsp+110h] [rbp-60h]
  int v100; // [rsp+118h] [rbp-58h]
  __int16 v101; // [rsp+11Ch] [rbp-54h]
  char v102; // [rsp+11Eh] [rbp-52h]
  __int64 v103; // [rsp+120h] [rbp-50h]
  __int64 v104; // [rsp+128h] [rbp-48h]
  void *v105; // [rsp+130h] [rbp-40h] BYREF
  void *v106; // [rsp+138h] [rbp-38h] BYREF

  v1 = (__int64)a1;
  v96 = (_QWORD *)sub_BD5C60((__int64)a1);
  v97 = &v105;
  v98 = &v106;
  v90 = (unsigned int *)v92;
  v105 = &unk_49DA100;
  v91 = 0x200000000LL;
  v99 = 0;
  v100 = 0;
  v101 = 512;
  v102 = 7;
  v103 = 0;
  v104 = 0;
  v93 = 0;
  v94 = 0;
  LOWORD(v95) = 0;
  v106 = &unk_49DA0B0;
  sub_D5F1F0((__int64)&v90, (__int64)a1);
  if ( *a1 != 52 )
    goto LABEL_2;
  v30 = *((_QWORD *)a1 - 8);
  v83 = *((_QWORD *)a1 - 4);
  v31 = *(_DWORD *)(*(_QWORD *)(v30 + 8) + 8LL) >> 8;
  v32 = sub_BCD140(v96, v31);
  v79 = (_BYTE *)sub_ACD640(v32, v31 - 1, 0);
  v87 = 257;
  v89 = 257;
  v33 = (unsigned __int8 *)sub_BD2C40(72, 1u);
  if ( v33 )
    sub_B549F0((__int64)v33, v30, (__int64)v88, 0, 0);
  (*((void (__fastcall **)(void **, unsigned __int8 *, _BYTE *, __int64, __int64))*v98 + 2))(v98, v33, v86, v94, v95);
  v34 = v90;
  v35 = &v90[4 * (unsigned int)v91];
  if ( v90 != v35 )
  {
    do
    {
      v36 = *((_QWORD *)v34 + 1);
      v37 = *v34;
      v34 += 4;
      sub_B99FD0((__int64)v33, v37, v36);
    }
    while ( v35 != v34 );
  }
  v87 = 257;
  v89 = 257;
  v38 = (unsigned __int8 *)sub_BD2C40(72, 1u);
  v39 = v38;
  if ( v38 )
    sub_B549F0((__int64)v38, v83, (__int64)v88, 0, 0);
  (*((void (__fastcall **)(void **, unsigned __int8 *, _BYTE *, __int64, __int64))*v98 + 2))(v98, v39, v86, v94, v95);
  v40 = 4LL * (unsigned int)v91;
  if ( v90 != &v90[v40] )
  {
    v41 = &v90[v40];
    v42 = v90;
    do
    {
      v43 = *((_QWORD *)v42 + 1);
      v44 = *v42;
      v42 += 4;
      sub_B99FD0((__int64)v39, v44, v43);
    }
    while ( v41 != v42 );
    v1 = (__int64)a1;
  }
  v89 = 257;
  v45 = sub_920F70(&v90, v33, v79, (__int64)v88, 0);
  v89 = 257;
  v84 = (unsigned __int8 *)v45;
  v46 = sub_920F70(&v90, v39, v79, (__int64)v88, 0);
  v87 = 257;
  v80 = (unsigned __int8 *)v46;
  v47 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v97 + 2);
  if ( v47 != sub_9202E0 )
  {
    v77 = (_BYTE *)v47((__int64)v97, 30u, v33, v84);
    goto LABEL_51;
  }
  if ( *v33 <= 0x15u && *v84 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(30) )
      v77 = (_BYTE *)sub_AD5570(30, (__int64)v33, v84, 0, 0);
    else
      v77 = (_BYTE *)sub_AABE40(0x1Eu, v33, v84);
LABEL_51:
    if ( v77 )
      goto LABEL_52;
  }
  v89 = 257;
  v77 = (_BYTE *)sub_B504D0(30, (__int64)v33, (__int64)v84, (__int64)v88, 0, 0);
  (*((void (__fastcall **)(void **, _BYTE *, _BYTE *, __int64, __int64))*v98 + 2))(v98, v77, v86, v94, v95);
  if ( v90 != &v90[4 * (unsigned int)v91] )
  {
    v76 = v1;
    v60 = v90;
    v61 = &v90[4 * (unsigned int)v91];
    do
    {
      v62 = *((_QWORD *)v60 + 1);
      v63 = *v60;
      v60 += 4;
      sub_B99FD0((__int64)v77, v63, v62);
    }
    while ( v61 != v60 );
    v1 = v76;
  }
LABEL_52:
  v87 = 257;
  v48 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v97 + 2);
  if ( v48 != sub_9202E0 )
  {
    v49 = (_BYTE *)v48((__int64)v97, 30u, v39, v80);
    goto LABEL_57;
  }
  if ( *v39 <= 0x15u && *v80 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(30) )
      v49 = (_BYTE *)sub_AD5570(30, (__int64)v39, v80, 0, 0);
    else
      v49 = (_BYTE *)sub_AABE40(0x1Eu, v39, v80);
LABEL_57:
    if ( v49 )
      goto LABEL_58;
  }
  v89 = 257;
  v49 = (_BYTE *)sub_B504D0(30, (__int64)v39, (__int64)v80, (__int64)v88, 0, 0);
  (*((void (__fastcall **)(void **, _BYTE *, _BYTE *, __int64, __int64))*v98 + 2))(v98, v49, v86, v94, v95);
  v64 = v90;
  v65 = &v90[4 * (unsigned int)v91];
  if ( v90 != v65 )
  {
    do
    {
      v66 = *((_QWORD *)v64 + 1);
      v67 = *v64;
      v64 += 4;
      sub_B99FD0((__int64)v49, v67, v66);
    }
    while ( v65 != v64 );
  }
LABEL_58:
  v89 = 257;
  v78 = (unsigned __int8 *)sub_929DE0(&v90, v77, v84, (__int64)v88, 0, 0);
  v89 = 257;
  v50 = (unsigned __int8 *)sub_929DE0(&v90, v49, v80, (__int64)v88, 0, 0);
  v87 = 257;
  v51 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v97 + 2);
  if ( v51 != sub_9202E0 )
  {
    v52 = (unsigned __int8 *)v51((__int64)v97, 22u, v78, v50);
    goto LABEL_63;
  }
  if ( *v78 <= 0x15u && *v50 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(22) )
      v52 = (unsigned __int8 *)sub_AD5570(22, (__int64)v78, v50, 0, 0);
    else
      v52 = (unsigned __int8 *)sub_AABE40(0x16u, v78, v50);
LABEL_63:
    if ( v52 )
      goto LABEL_64;
  }
  v89 = 257;
  v52 = (unsigned __int8 *)sub_B504D0(22, (__int64)v78, (__int64)v50, (__int64)v88, 0, 0);
  (*((void (__fastcall **)(void **, unsigned __int8 *, _BYTE *, __int64, __int64))*v98 + 2))(v98, v52, v86, v94, v95);
  v72 = v90;
  v73 = &v90[4 * (unsigned int)v91];
  if ( v90 != v73 )
  {
    do
    {
      v74 = *((_QWORD *)v72 + 1);
      v75 = *v72;
      v72 += 4;
      sub_B99FD0((__int64)v52, v75, v74);
    }
    while ( v73 != v72 );
  }
LABEL_64:
  v87 = 257;
  v53 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v97 + 2);
  if ( v53 != sub_9202E0 )
  {
    v54 = (_BYTE *)v53((__int64)v97, 30u, v52, v84);
    goto LABEL_69;
  }
  if ( *v52 <= 0x15u && *v84 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(30) )
      v54 = (_BYTE *)sub_AD5570(30, (__int64)v52, v84, 0, 0);
    else
      v54 = (_BYTE *)sub_AABE40(0x1Eu, v52, v84);
LABEL_69:
    if ( v54 )
      goto LABEL_70;
  }
  v89 = 257;
  v54 = (_BYTE *)sub_B504D0(30, (__int64)v52, (__int64)v84, (__int64)v88, 0, 0);
  (*((void (__fastcall **)(void **, _BYTE *, _BYTE *, __int64, __int64))*v98 + 2))(v98, v54, v86, v94, v95);
  if ( v90 != &v90[4 * (unsigned int)v91] )
  {
    v81 = v1;
    v68 = v90;
    v69 = &v90[4 * (unsigned int)v91];
    do
    {
      v70 = *((_QWORD *)v68 + 1);
      v71 = *v68;
      v68 += 4;
      sub_B99FD0((__int64)v54, v71, v70);
    }
    while ( v69 != v68 );
    v1 = v81;
  }
LABEL_70:
  v89 = 257;
  v55 = sub_929DE0(&v90, v54, v84, (__int64)v88, 0, 0);
  if ( *v52 > 0x1Cu )
    sub_D5F1F0((__int64)&v90, (__int64)v52);
  v56 = v94;
  sub_BD84D0(v1, v55);
  if ( (*(_BYTE *)(v1 + 7) & 0x40) != 0 )
  {
    v57 = *(_QWORD *)(v1 - 8);
    v58 = v57 + 32LL * (*(_DWORD *)(v1 + 4) & 0x7FFFFFF);
  }
  else
  {
    v58 = v1;
    v57 = v1 - 32LL * (*(_DWORD *)(v1 + 4) & 0x7FFFFFF);
  }
  for ( ; v58 != v57; v57 += 32 )
  {
    if ( *(_QWORD *)v57 )
    {
      v59 = *(_QWORD *)(v57 + 8);
      **(_QWORD **)(v57 + 16) = v59;
      if ( v59 )
        *(_QWORD *)(v59 + 16) = *(_QWORD *)(v57 + 16);
    }
    *(_QWORD *)v57 = 0;
  }
  sub_B43D60((_QWORD *)v1);
  if ( v1 + 24 != v56 )
  {
    if ( !v94 )
      BUG();
    v1 = v94 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v94 - 24) - 42 > 0x11 )
      BUG();
LABEL_2:
    v2 = *(_QWORD *)(v1 - 32);
    v3 = *(_QWORD *)(v1 - 64);
    v87 = 257;
    v82 = v2;
    v89 = 257;
    v4 = (unsigned __int8 *)sub_BD2C40(72, 1u);
    v5 = v4;
    if ( v4 )
      sub_B549F0((__int64)v4, v3, (__int64)v88, 0, 0);
    (*((void (__fastcall **)(void **, unsigned __int8 *, _BYTE *, __int64, __int64))*v98 + 2))(v98, v5, v86, v94, v95);
    v6 = v90;
    v7 = &v90[4 * (unsigned int)v91];
    if ( v90 != v7 )
    {
      do
      {
        v8 = *((_QWORD *)v6 + 1);
        v9 = *v6;
        v6 += 4;
        sub_B99FD0((__int64)v5, v9, v8);
      }
      while ( v7 != v6 );
    }
    v87 = 257;
    v89 = 257;
    v10 = (unsigned __int8 *)sub_BD2C40(72, 1u);
    v11 = v10;
    if ( v10 )
      sub_B549F0((__int64)v10, v82, (__int64)v88, 0, 0);
    (*((void (__fastcall **)(void **, unsigned __int8 *, _BYTE *, __int64, __int64))*v98 + 2))(v98, v11, v86, v94, v95);
    v12 = v90;
    v13 = &v90[4 * (unsigned int)v91];
    if ( v90 != v13 )
    {
      do
      {
        v14 = *((_QWORD *)v12 + 1);
        v15 = *v12;
        v12 += 4;
        sub_B99FD0((__int64)v11, v15, v14);
      }
      while ( v13 != v12 );
    }
    v89 = 257;
    v16 = sub_3122580((__int64 *)&v90, v5, v11, (__int64)v88, 0);
    v87 = 257;
    v17 = (unsigned __int8 *)v16;
    v18 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))*((_QWORD *)*v97 + 4);
    if ( v18 == sub_9201A0 )
    {
      if ( *v11 > 0x15u || *v17 > 0x15u )
        goto LABEL_31;
      if ( (unsigned __int8)sub_AC47B0(17) )
        v19 = (_BYTE *)sub_AD5570(17, (__int64)v11, v17, 0, 0);
      else
        v19 = (_BYTE *)sub_AABE40(0x11u, v11, v17);
    }
    else
    {
      v19 = (_BYTE *)v18((__int64)v97, 17u, v11, v17, 0, 0);
    }
    if ( v19 )
    {
LABEL_16:
      v89 = 257;
      v20 = sub_929DE0(&v90, v5, v19, (__int64)v88, 0, 0);
      if ( *v17 > 0x1Cu )
        sub_D5F1F0((__int64)&v90, (__int64)v17);
      sub_BD84D0(v1, v20);
      if ( (*(_BYTE *)(v1 + 7) & 0x40) != 0 )
      {
        v21 = *(_QWORD *)(v1 - 8);
        v22 = v21 + 32LL * (*(_DWORD *)(v1 + 4) & 0x7FFFFFF);
      }
      else
      {
        v22 = v1;
        v21 = v1 - 32LL * (*(_DWORD *)(v1 + 4) & 0x7FFFFFF);
      }
      for ( ; v22 != v21; v21 += 32 )
      {
        if ( *(_QWORD *)v21 )
        {
          v23 = *(_QWORD *)(v21 + 8);
          **(_QWORD **)(v21 + 16) = v23;
          if ( v23 )
            *(_QWORD *)(v23 + 16) = *(_QWORD *)(v21 + 16);
        }
        *(_QWORD *)v21 = 0;
      }
      sub_B43D60((_QWORD *)v1);
      if ( !v94 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v94 - 24) - 42 <= 0x11 )
        sub_3198EC0((_BYTE *)(v94 - 24));
      goto LABEL_28;
    }
LABEL_31:
    v89 = 257;
    v19 = (_BYTE *)sub_B504D0(17, (__int64)v11, (__int64)v17, (__int64)v88, 0, 0);
    (*((void (__fastcall **)(void **, _BYTE *, _BYTE *, __int64, __int64))*v98 + 2))(v98, v19, v86, v94, v95);
    v25 = 4LL * (unsigned int)v91;
    v26 = &v90[v25];
    if ( v90 != &v90[v25] )
    {
      v85 = v1;
      v27 = v90;
      do
      {
        v28 = *((_QWORD *)v27 + 1);
        v29 = *v27;
        v27 += 4;
        sub_B99FD0((__int64)v19, v29, v28);
      }
      while ( v26 != v27 );
      v1 = v85;
    }
    goto LABEL_16;
  }
LABEL_28:
  nullsub_61();
  v105 = &unk_49DA100;
  nullsub_63();
  if ( v90 != (unsigned int *)v92 )
    _libc_free((unsigned __int64)v90);
  return 1;
}
