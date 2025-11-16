// Function: sub_2438890
// Address: 0x2438890
//
unsigned __int64 *__fastcall sub_2438890(
        unsigned __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  __int64 v8; // rbx
  __int64 v9; // rax
  _BYTE *v10; // rax
  __int64 v11; // rsi
  __int64 v12; // r13
  __int64 **v13; // rax
  unsigned __int8 *v14; // r12
  __int64 (__fastcall *v15)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8); // rax
  __int64 v16; // r15
  __int64 (__fastcall *v17)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v18; // rdx
  int v19; // esi
  char v20; // di
  __int64 v21; // rax
  unsigned __int64 v22; // rax
  __int64 v23; // r13
  __int64 v24; // r15
  __int64 v25; // rax
  char v26; // al
  _QWORD *v27; // rax
  __int64 v28; // r9
  _BYTE *v29; // r12
  unsigned int *v30; // r15
  unsigned int *v31; // r13
  __int64 v32; // rdx
  unsigned int v33; // esi
  unsigned __int8 *v34; // r13
  __int64 v35; // rax
  __int64 v37; // r12
  unsigned int *v38; // rcx
  unsigned int *v39; // r12
  __int64 v40; // r14
  unsigned int *v41; // rbx
  __int64 v42; // rdx
  unsigned int v43; // esi
  _BYTE *v44; // rax
  __int64 v45; // rax
  unsigned __int8 *v46; // r12
  __int64 (__fastcall *v47)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v48; // r14
  unsigned int *v49; // r15
  unsigned int *v50; // rbx
  __int64 v51; // rdx
  unsigned int v52; // esi
  unsigned int *v53; // r13
  unsigned int *v54; // r12
  __int64 v55; // rdx
  unsigned int v56; // esi
  __int64 v57; // [rsp-10h] [rbp-1C0h]
  char v58; // [rsp+4h] [rbp-1ACh]
  __int64 **v62; // [rsp+40h] [rbp-170h]
  __int64 v63; // [rsp+40h] [rbp-170h]
  __int64 v64; // [rsp+48h] [rbp-168h]
  _BYTE v66[32]; // [rsp+60h] [rbp-150h] BYREF
  __int16 v67; // [rsp+80h] [rbp-130h]
  _BYTE v68[32]; // [rsp+90h] [rbp-120h] BYREF
  __int16 v69; // [rsp+B0h] [rbp-100h]
  _QWORD v70[4]; // [rsp+C0h] [rbp-F0h] BYREF
  __int16 v71; // [rsp+E0h] [rbp-D0h]
  unsigned int *v72; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v73; // [rsp+F8h] [rbp-B8h]
  _BYTE v74[32]; // [rsp+100h] [rbp-B0h] BYREF
  __int64 v75; // [rsp+120h] [rbp-90h]
  __int64 v76; // [rsp+128h] [rbp-88h]
  __int64 v77; // [rsp+130h] [rbp-80h]
  __int64 v78; // [rsp+138h] [rbp-78h]
  void **v79; // [rsp+140h] [rbp-70h]
  void **v80; // [rsp+148h] [rbp-68h]
  __int64 v81; // [rsp+150h] [rbp-60h]
  int v82; // [rsp+158h] [rbp-58h]
  __int16 v83; // [rsp+15Ch] [rbp-54h]
  char v84; // [rsp+15Eh] [rbp-52h]
  __int64 v85; // [rsp+160h] [rbp-50h]
  __int64 v86; // [rsp+168h] [rbp-48h]
  void *v87; // [rsp+170h] [rbp-40h] BYREF
  void *v88; // [rsp+178h] [rbp-38h] BYREF

  v8 = a2;
  *a1 = 0;
  v9 = sub_BD5C60(a4);
  v84 = 7;
  v78 = v9;
  v79 = &v87;
  v80 = &v88;
  v83 = 512;
  LOWORD(v77) = 0;
  v72 = (unsigned int *)v74;
  v87 = &unk_49DA100;
  v73 = 0x200000000LL;
  v81 = 0;
  v88 = &unk_49DA0B0;
  v82 = 0;
  v85 = 0;
  v86 = 0;
  v75 = 0;
  v76 = 0;
  sub_D5F1F0((__int64)&v72, a4);
  v71 = 257;
  v10 = sub_94BCF0(&v72, a3, *(_QWORD *)(a2 + 120), (__int64)v70);
  v11 = *(unsigned int *)(a2 + 176);
  v12 = (__int64)v10;
  v69 = 257;
  a1[1] = (unsigned __int64)v10;
  v13 = *(__int64 ***)(v8 + 136);
  v67 = 257;
  v62 = v13;
  v14 = (unsigned __int8 *)sub_AD64C0(*(_QWORD *)(v12 + 8), v11, 0);
  v15 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8))*((_QWORD *)*v79 + 3);
  if ( v15 == sub_920250 )
  {
    if ( *(_BYTE *)v12 > 0x15u || *v14 > 0x15u )
      goto LABEL_21;
    if ( (unsigned __int8)sub_AC47B0(26) )
      v16 = sub_AD5570(26, v12, v14, 0, 0);
    else
      v16 = sub_AABE40(0x1Au, (unsigned __int8 *)v12, v14);
  }
  else
  {
    v16 = v15((__int64)v79, 26u, (_BYTE *)v12, v14, 0);
  }
  if ( v16 )
    goto LABEL_7;
LABEL_21:
  v71 = 257;
  v16 = sub_B504D0(26, v12, (__int64)v14, (__int64)v70, 0, 0);
  (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v80 + 2))(v80, v16, v66, v76, v77);
  v37 = 4LL * (unsigned int)v73;
  v38 = &v72[v37];
  if ( v72 == &v72[v37] )
  {
LABEL_7:
    if ( v62 != *(__int64 ***)(v16 + 8) )
      goto LABEL_8;
LABEL_25:
    v64 = v16;
    goto LABEL_13;
  }
  v39 = v72;
  v40 = v8;
  v41 = v38;
  do
  {
    v42 = *((_QWORD *)v39 + 1);
    v43 = *v39;
    v39 += 4;
    sub_B99FD0(v16, v43, v42);
  }
  while ( v41 != v39 );
  v8 = v40;
  if ( v62 == *(__int64 ***)(v16 + 8) )
    goto LABEL_25;
LABEL_8:
  v17 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, __int64))*((_QWORD *)*v79 + 15);
  if ( v17 != sub_920130 )
  {
    v64 = v17((__int64)v79, 38u, (_BYTE *)v16, (__int64)v62);
    goto LABEL_12;
  }
  if ( *(_BYTE *)v16 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x26u) )
      v64 = sub_ADAB70(38, v16, v62, 0);
    else
      v64 = sub_AA93C0(0x26u, v16, (__int64)v62);
LABEL_12:
    if ( v64 )
      goto LABEL_13;
  }
  v71 = 257;
  v64 = sub_B51D30(38, v16, (__int64)v62, (__int64)v70, 0, 0);
  (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v80 + 2))(v80, v64, v68, v76, v77);
  v49 = &v72[4 * (unsigned int)v73];
  if ( v72 != v49 )
  {
    v63 = v8;
    v50 = v72;
    do
    {
      v51 = *((_QWORD *)v50 + 1);
      v52 = *v50;
      v50 += 4;
      sub_B99FD0(v64, v52, v51);
    }
    while ( v49 != v50 );
    v8 = v63;
  }
LABEL_13:
  v18 = *(_QWORD *)(v8 + 184);
  v19 = *(_DWORD *)(v8 + 176);
  v20 = *(_BYTE *)(v8 + 160);
  a1[3] = v64;
  v21 = sub_2435400(v20, v19, v18, (__int64 *)&v72, v12);
  a1[2] = v21;
  v22 = sub_2436FF0(v8, v21, (__int64)&v72);
  v23 = *(_QWORD *)(v8 + 136);
  v69 = 257;
  v24 = v22;
  v25 = sub_AA4E30(v75);
  v26 = sub_AE5020(v25, v23);
  v71 = 257;
  v58 = v26;
  v27 = sub_BD2C40(80, unk_3F10A14);
  v29 = v27;
  if ( v27 )
  {
    sub_B4D190((__int64)v27, v23, v24, (__int64)v70, 0, v58, 0, 0);
    v28 = v57;
  }
  (*((void (__fastcall **)(void **, _BYTE *, _BYTE *, __int64, __int64, __int64))*v80 + 2))(
    v80,
    v29,
    v68,
    v76,
    v77,
    v28);
  v30 = v72;
  v31 = &v72[4 * (unsigned int)v73];
  if ( v72 != v31 )
  {
    do
    {
      v32 = *((_QWORD *)v30 + 1);
      v33 = *v30;
      v30 += 4;
      sub_B99FD0((__int64)v29, v33, v32);
    }
    while ( v31 != v30 );
  }
  v71 = 257;
  a1[4] = (unsigned __int64)v29;
  v34 = (unsigned __int8 *)sub_92B530(&v72, 0x21u, v64, v29, (__int64)v70);
  if ( *(_BYTE *)(v8 + 173) )
  {
    v71 = 257;
    v44 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v64 + 8), *(unsigned __int8 *)(v8 + 172), 0);
    v45 = sub_92B530(&v72, 0x21u, v64, v44, (__int64)v70);
    v69 = 257;
    v46 = (unsigned __int8 *)v45;
    v47 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v79 + 2);
    if ( v47 == sub_9202E0 )
    {
      if ( *v34 > 0x15u || *v46 > 0x15u )
        goto LABEL_39;
      if ( (unsigned __int8)sub_AC47B0(28) )
        v48 = sub_AD5570(28, (__int64)v34, v46, 0, 0);
      else
        v48 = sub_AABE40(0x1Cu, v34, v46);
    }
    else
    {
      v48 = v47((__int64)v79, 28u, v34, v46);
    }
    if ( v48 )
    {
LABEL_32:
      v34 = (unsigned __int8 *)v48;
      goto LABEL_18;
    }
LABEL_39:
    v71 = 257;
    v48 = sub_B504D0(28, (__int64)v34, (__int64)v46, (__int64)v70, 0, 0);
    (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v80 + 2))(v80, v48, v68, v76, v77);
    v53 = v72;
    v54 = &v72[4 * (unsigned int)v73];
    if ( v72 != v54 )
    {
      do
      {
        v55 = *((_QWORD *)v53 + 1);
        v56 = *v53;
        v53 += 4;
        sub_B99FD0(v48, v56, v55);
      }
      while ( v54 != v53 );
    }
    goto LABEL_32;
  }
LABEL_18:
  v70[0] = *(_QWORD *)v8;
  v35 = sub_B8C340(v70);
  *a1 = sub_F38250((__int64)v34, (__int64 *)(a4 + 24), 0, 0, v35, a5, a6, 0);
  nullsub_61();
  v87 = &unk_49DA100;
  nullsub_63();
  if ( v72 != (unsigned int *)v74 )
    _libc_free((unsigned __int64)v72);
  return a1;
}
