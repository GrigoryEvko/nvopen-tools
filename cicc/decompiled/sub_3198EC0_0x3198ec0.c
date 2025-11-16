// Function: sub_3198EC0
// Address: 0x3198ec0
//
__int64 __fastcall sub_3198EC0(_BYTE *a1)
{
  __int64 v1; // r12
  __int64 v2; // rax
  __int64 v3; // rcx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v7; // rbx
  __int64 v8; // r15
  unsigned int v9; // r13d
  __int64 v10; // rax
  unsigned __int8 *v11; // r14
  unsigned int *v12; // rbx
  unsigned int *v13; // r13
  __int64 v14; // rdx
  unsigned int v15; // esi
  unsigned __int8 *v16; // rax
  unsigned __int8 *v17; // r13
  unsigned int *v18; // r15
  unsigned int *v19; // rbx
  __int64 v20; // rdx
  unsigned int v21; // esi
  _BYTE *v22; // r15
  unsigned __int8 *v23; // r15
  __int64 (__fastcall *v24)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  _BYTE *v25; // rbx
  __int64 v26; // rax
  __int64 (__fastcall *v27)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  _BYTE *v28; // r14
  __int64 v29; // rax
  unsigned __int8 *v30; // r14
  __int64 (__fastcall *v31)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  unsigned __int8 *v32; // r13
  __int64 v33; // rax
  unsigned __int8 *v34; // r14
  __int64 (__fastcall *v35)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  _BYTE *v36; // rbx
  __int64 v37; // r15
  __int64 v38; // r13
  __int64 v39; // rax
  __int64 v40; // rcx
  __int64 v41; // rdx
  unsigned int *v42; // r12
  unsigned int *v43; // r14
  __int64 v44; // rdx
  unsigned int v45; // esi
  unsigned int *v46; // r12
  unsigned int *v47; // r15
  __int64 v48; // rdx
  unsigned int v49; // esi
  unsigned int *v50; // r15
  unsigned int *v51; // rbx
  __int64 v52; // rdx
  unsigned int v53; // esi
  unsigned int *v54; // r13
  unsigned int *v55; // rbx
  __int64 v56; // rdx
  unsigned int v57; // esi
  unsigned __int8 *v58; // [rsp+0h] [rbp-160h]
  _BYTE *v59; // [rsp+10h] [rbp-150h]
  unsigned __int8 *v60; // [rsp+10h] [rbp-150h]
  __int64 v61; // [rsp+10h] [rbp-150h]
  _BYTE v62[32]; // [rsp+40h] [rbp-120h] BYREF
  __int16 v63; // [rsp+60h] [rbp-100h]
  _BYTE v64[32]; // [rsp+70h] [rbp-F0h] BYREF
  __int16 v65; // [rsp+90h] [rbp-D0h]
  unsigned int *v66; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v67; // [rsp+A8h] [rbp-B8h]
  _BYTE v68[32]; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v69; // [rsp+D0h] [rbp-90h]
  __int64 v70; // [rsp+D8h] [rbp-88h]
  __int64 v71; // [rsp+E0h] [rbp-80h]
  _QWORD *v72; // [rsp+E8h] [rbp-78h]
  void **v73; // [rsp+F0h] [rbp-70h]
  void **v74; // [rsp+F8h] [rbp-68h]
  __int64 v75; // [rsp+100h] [rbp-60h]
  int v76; // [rsp+108h] [rbp-58h]
  __int16 v77; // [rsp+10Ch] [rbp-54h]
  char v78; // [rsp+10Eh] [rbp-52h]
  __int64 v79; // [rsp+110h] [rbp-50h]
  __int64 v80; // [rsp+118h] [rbp-48h]
  void *v81; // [rsp+120h] [rbp-40h] BYREF
  void *v82; // [rsp+128h] [rbp-38h] BYREF

  v1 = (__int64)a1;
  v78 = 7;
  v72 = (_QWORD *)sub_BD5C60((__int64)a1);
  v73 = &v81;
  v74 = &v82;
  v81 = &unk_49DA100;
  v66 = (unsigned int *)v68;
  v67 = 0x200000000LL;
  v75 = 0;
  v76 = 0;
  v77 = 512;
  v79 = 0;
  v80 = 0;
  v69 = 0;
  v70 = 0;
  LOWORD(v71) = 0;
  v82 = &unk_49DA0B0;
  sub_D5F1F0((__int64)&v66, (__int64)a1);
  if ( *a1 != 49 )
  {
LABEL_2:
    v2 = sub_3196910(*(_QWORD *)(v1 - 64), *(_QWORD *)(v1 - 32), (__int64)&v66);
    sub_BD84D0(v1, v2);
    if ( (*(_BYTE *)(v1 + 7) & 0x40) != 0 )
    {
      v4 = *(_QWORD *)(v1 - 8);
      v3 = v4 + 32LL * (*(_DWORD *)(v1 + 4) & 0x7FFFFFF);
    }
    else
    {
      v3 = v1;
      v4 = v1 - 32LL * (*(_DWORD *)(v1 + 4) & 0x7FFFFFF);
    }
    for ( ; v3 != v4; v4 += 32 )
    {
      if ( *(_QWORD *)v4 )
      {
        v5 = *(_QWORD *)(v4 + 8);
        **(_QWORD **)(v4 + 16) = v5;
        if ( v5 )
          *(_QWORD *)(v5 + 16) = *(_QWORD *)(v4 + 16);
      }
      *(_QWORD *)v4 = 0;
    }
    sub_B43D60((_QWORD *)v1);
    goto LABEL_10;
  }
  v7 = *((_QWORD *)a1 - 8);
  v8 = *((_QWORD *)a1 - 4);
  v9 = *(_DWORD *)(*(_QWORD *)(v7 + 8) + 8LL) >> 8;
  v10 = sub_BCD140(v72, v9);
  v59 = (_BYTE *)sub_ACD640(v10, v9 - 1, 0);
  v65 = 257;
  v63 = 257;
  v11 = (unsigned __int8 *)sub_BD2C40(72, 1u);
  if ( v11 )
    sub_B549F0((__int64)v11, v7, (__int64)v64, 0, 0);
  (*((void (__fastcall **)(void **, unsigned __int8 *, _BYTE *, __int64, __int64))*v74 + 2))(v74, v11, v62, v70, v71);
  v12 = v66;
  v13 = &v66[4 * (unsigned int)v67];
  if ( v66 != v13 )
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
  v63 = 257;
  v65 = 257;
  v16 = (unsigned __int8 *)sub_BD2C40(72, 1u);
  v17 = v16;
  if ( v16 )
    sub_B549F0((__int64)v16, v8, (__int64)v64, 0, 0);
  (*((void (__fastcall **)(void **, unsigned __int8 *, _BYTE *, __int64, __int64))*v74 + 2))(v74, v17, v62, v70, v71);
  v18 = v66;
  v19 = &v66[4 * (unsigned int)v67];
  if ( v66 != v19 )
  {
    do
    {
      v20 = *((_QWORD *)v18 + 1);
      v21 = *v18;
      v18 += 4;
      sub_B99FD0((__int64)v17, v21, v20);
    }
    while ( v19 != v18 );
  }
  v22 = v59;
  v65 = 257;
  v60 = (unsigned __int8 *)sub_920F70(&v66, v11, v59, (__int64)v64, 0);
  v65 = 257;
  v23 = (unsigned __int8 *)sub_920F70(&v66, v17, v22, (__int64)v64, 0);
  v63 = 257;
  v24 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v73 + 2);
  if ( v24 != sub_9202E0 )
  {
    v25 = (_BYTE *)v24((__int64)v73, 30u, v60, v11);
    goto LABEL_27;
  }
  if ( *v60 <= 0x15u && *v11 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(30) )
      v25 = (_BYTE *)sub_AD5570(30, (__int64)v60, v11, 0, 0);
    else
      v25 = (_BYTE *)sub_AABE40(0x1Eu, v60, v11);
LABEL_27:
    if ( v25 )
      goto LABEL_28;
  }
  v65 = 257;
  v25 = (_BYTE *)sub_B504D0(30, (__int64)v60, (__int64)v11, (__int64)v64, 0, 0);
  (*((void (__fastcall **)(void **, _BYTE *, _BYTE *, __int64, __int64))*v74 + 2))(v74, v25, v62, v70, v71);
  if ( v66 != &v66[4 * (unsigned int)v67] )
  {
    v42 = v66;
    v43 = &v66[4 * (unsigned int)v67];
    do
    {
      v44 = *((_QWORD *)v42 + 1);
      v45 = *v42;
      v42 += 4;
      sub_B99FD0((__int64)v25, v45, v44);
    }
    while ( v43 != v42 );
    v1 = (__int64)a1;
  }
LABEL_28:
  v65 = 257;
  v26 = sub_929DE0(&v66, v25, v60, (__int64)v64, 0, 0);
  v63 = 257;
  v58 = (unsigned __int8 *)v26;
  v27 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v73 + 2);
  if ( v27 != sub_9202E0 )
  {
    v28 = (_BYTE *)v27((__int64)v73, 30u, v23, v17);
    goto LABEL_33;
  }
  if ( *v23 <= 0x15u && *v17 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(30) )
      v28 = (_BYTE *)sub_AD5570(30, (__int64)v23, v17, 0, 0);
    else
      v28 = (_BYTE *)sub_AABE40(0x1Eu, v23, v17);
LABEL_33:
    if ( v28 )
      goto LABEL_34;
  }
  v65 = 257;
  v28 = (_BYTE *)sub_B504D0(30, (__int64)v23, (__int64)v17, (__int64)v64, 0, 0);
  (*((void (__fastcall **)(void **, _BYTE *, _BYTE *, __int64, __int64))*v74 + 2))(v74, v28, v62, v70, v71);
  v54 = v66;
  v55 = &v66[4 * (unsigned int)v67];
  if ( v66 != v55 )
  {
    do
    {
      v56 = *((_QWORD *)v54 + 1);
      v57 = *v54;
      v54 += 4;
      sub_B99FD0((__int64)v28, v57, v56);
    }
    while ( v55 != v54 );
  }
LABEL_34:
  v65 = 257;
  v29 = sub_929DE0(&v66, v28, v23, (__int64)v64, 0, 0);
  v63 = 257;
  v30 = (unsigned __int8 *)v29;
  v31 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v73 + 2);
  if ( v31 != sub_9202E0 )
  {
    v32 = (unsigned __int8 *)v31((__int64)v73, 30u, v23, v60);
    goto LABEL_39;
  }
  if ( *v23 <= 0x15u && *v60 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(30) )
      v32 = (unsigned __int8 *)sub_AD5570(30, (__int64)v23, v60, 0, 0);
    else
      v32 = (unsigned __int8 *)sub_AABE40(0x1Eu, v23, v60);
LABEL_39:
    if ( v32 )
      goto LABEL_40;
  }
  v65 = 257;
  v32 = (unsigned __int8 *)sub_B504D0(30, (__int64)v23, (__int64)v60, (__int64)v64, 0, 0);
  (*((void (__fastcall **)(void **, unsigned __int8 *, _BYTE *, __int64, __int64))*v74 + 2))(v74, v32, v62, v70, v71);
  v50 = v66;
  v51 = &v66[4 * (unsigned int)v67];
  if ( v66 != v51 )
  {
    do
    {
      v52 = *((_QWORD *)v50 + 1);
      v53 = *v50;
      v50 += 4;
      sub_B99FD0((__int64)v32, v53, v52);
    }
    while ( v51 != v50 );
  }
LABEL_40:
  v65 = 257;
  v33 = sub_3122580((__int64 *)&v66, v58, v30, (__int64)v64, 0);
  v63 = 257;
  v34 = (unsigned __int8 *)v33;
  v35 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v73 + 2);
  if ( v35 != sub_9202E0 )
  {
    v36 = (_BYTE *)v35((__int64)v73, 30u, v34, v32);
    goto LABEL_45;
  }
  if ( *v34 <= 0x15u && *v32 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(30) )
      v36 = (_BYTE *)sub_AD5570(30, (__int64)v34, v32, 0, 0);
    else
      v36 = (_BYTE *)sub_AABE40(0x1Eu, v34, v32);
LABEL_45:
    if ( v36 )
      goto LABEL_46;
  }
  v65 = 257;
  v36 = (_BYTE *)sub_B504D0(30, (__int64)v34, (__int64)v32, (__int64)v64, 0, 0);
  (*((void (__fastcall **)(void **, _BYTE *, _BYTE *, __int64, __int64))*v74 + 2))(v74, v36, v62, v70, v71);
  if ( v66 != &v66[4 * (unsigned int)v67] )
  {
    v61 = v1;
    v46 = v66;
    v47 = &v66[4 * (unsigned int)v67];
    do
    {
      v48 = *((_QWORD *)v46 + 1);
      v49 = *v46;
      v46 += 4;
      sub_B99FD0((__int64)v36, v49, v48);
    }
    while ( v47 != v46 );
    v1 = v61;
  }
LABEL_46:
  v65 = 257;
  v37 = sub_929DE0(&v66, v36, v32, (__int64)v64, 0, 0);
  if ( *v34 > 0x1Cu )
    sub_D5F1F0((__int64)&v66, (__int64)v34);
  v38 = v70;
  sub_BD84D0(v1, v37);
  if ( (*(_BYTE *)(v1 + 7) & 0x40) != 0 )
  {
    v39 = *(_QWORD *)(v1 - 8);
    v40 = v39 + 32LL * (*(_DWORD *)(v1 + 4) & 0x7FFFFFF);
  }
  else
  {
    v40 = v1;
    v39 = v1 - 32LL * (*(_DWORD *)(v1 + 4) & 0x7FFFFFF);
  }
  for ( ; v39 != v40; v39 += 32 )
  {
    if ( *(_QWORD *)v39 )
    {
      v41 = *(_QWORD *)(v39 + 8);
      **(_QWORD **)(v39 + 16) = v41;
      if ( v41 )
        *(_QWORD *)(v41 + 16) = *(_QWORD *)(v39 + 16);
    }
    *(_QWORD *)v39 = 0;
  }
  sub_B43D60((_QWORD *)v1);
  if ( v1 + 24 != v38 )
  {
    if ( !v70 )
      BUG();
    v1 = v70 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v70 - 24) - 42 > 0x11 )
      BUG();
    goto LABEL_2;
  }
LABEL_10:
  nullsub_61();
  v81 = &unk_49DA100;
  nullsub_63();
  if ( v66 != (unsigned int *)v68 )
    _libc_free((unsigned __int64)v66);
  return 1;
}
