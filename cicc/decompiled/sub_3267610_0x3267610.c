// Function: sub_3267610
// Address: 0x3267610
//
bool __fastcall sub_3267610(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r15
  unsigned __int64 v5; // r14
  __int64 *v6; // r13
  __int16 v7; // dx
  __int16 v8; // cx
  bool v9; // r15
  __int16 v10; // ax
  __int16 v11; // dx
  char v12; // cl
  char v13; // si
  __int64 v14; // rbx
  unsigned __int64 v15; // rsi
  char *v16; // rax
  char *v17; // r12
  char *v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // rdx
  char *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  _QWORD *v25; // r9
  __int64 *v26; // r10
  unsigned __int64 v27; // r11
  __int64 v28; // rax
  _QWORD *v29; // rdi
  char v30; // al
  __int64 *v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned __int64 v34; // rax
  __int64 v35; // rsi
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // rsi
  __int64 v39; // rcx
  __int64 v40; // rdx
  __int64 v41; // r8
  __int64 v42; // rsi
  __int64 v43; // rcx
  __int64 v44; // rdx
  __int64 v45; // rax
  unsigned __int64 v46; // rax
  __int64 v48; // rdi
  unsigned __int64 v49; // r8
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 v52; // [rsp+0h] [rbp-130h]
  unsigned __int64 v53; // [rsp+8h] [rbp-128h]
  bool v54; // [rsp+17h] [rbp-119h]
  unsigned __int64 v55; // [rsp+18h] [rbp-118h]
  __int64 *v56; // [rsp+18h] [rbp-118h]
  __int64 v57; // [rsp+20h] [rbp-110h]
  __int64 v58; // [rsp+20h] [rbp-110h]
  _QWORD *v59; // [rsp+20h] [rbp-110h]
  char v61; // [rsp+3Fh] [rbp-F1h] BYREF
  _BYTE v62[8]; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v63; // [rsp+48h] [rbp-E8h]
  int v64; // [rsp+50h] [rbp-E0h]
  __int64 v65; // [rsp+58h] [rbp-D8h]
  unsigned __int64 v66; // [rsp+60h] [rbp-D0h]
  __int64 *v67; // [rsp+68h] [rbp-C8h]
  _BYTE v68[8]; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v69; // [rsp+78h] [rbp-B8h]
  int v70; // [rsp+80h] [rbp-B0h]
  __int64 v71; // [rsp+88h] [rbp-A8h]
  unsigned __int64 v72; // [rsp+90h] [rbp-A0h]
  __int64 v73; // [rsp+98h] [rbp-98h]
  _QWORD v74[6]; // [rsp+A0h] [rbp-90h] BYREF
  unsigned __int64 v75; // [rsp+D0h] [rbp-60h] BYREF
  unsigned __int64 v76; // [rsp+D8h] [rbp-58h]
  __int64 v77; // [rsp+E0h] [rbp-50h]
  __int64 v78; // [rsp+E8h] [rbp-48h]
  __int64 v79; // [rsp+F0h] [rbp-40h]
  __int64 v80; // [rsp+F8h] [rbp-38h]

  sub_3267350((__int64)v62, a2);
  v4 = v63;
  v5 = v66;
  v6 = v67;
  v57 = v65;
  sub_3267350((__int64)v68, a3);
  if ( v4 && v4 == v69 && v70 == v64 && v71 == v57 || v62[0] && v68[0] || v62[1] && v68[1] )
    return 1;
  if ( v73 && v6 )
  {
    v7 = *((_WORD *)v6 + 16);
    v8 = *(_WORD *)(v73 + 32);
    if ( (v7 & 0x20) != 0 && (v8 & 2) != 0 )
      return 0;
    if ( (v8 & 0x20) != 0 && (v7 & 2) != 0 )
      return 0;
  }
  v54 = v5 != 0xBFFFFFFFFFFFFFFELL && v5 != -1;
  if ( v54 && (v5 & 0x4000000000000000LL) != 0 && v57 )
    return 1;
  v9 = v72 != -1 && v72 != 0xBFFFFFFFFFFFFFFELL;
  if ( v9 && (v72 & 0x4000000000000000LL) != 0 && v71 )
    return 1;
  v55 = v72;
  v58 = v73;
  if ( (unsigned __int8)sub_3364B70(a2, v66, a3, v72, *a1, &v61) )
    return v61;
  if ( !v58 || !v6 )
    return 1;
  v10 = *((_WORD *)v6 + 16);
  v11 = *(_WORD *)(v58 + 32);
  if ( (v10 & 0x20) != 0 && (v11 & 2) != 0 )
    return 0;
  if ( (v11 & 0x20) != 0 && (v10 & 2) != 0 )
    return 0;
  v12 = *((_BYTE *)v6 + 34);
  v13 = *(_BYTE *)(v58 + 34);
  v14 = v6[1];
  v52 = *(_QWORD *)(v58 + 8);
  if ( v52 == v14 || v13 != v12 || !v54 )
    goto LABEL_26;
  if ( !v9 )
    goto LABEL_26;
  if ( ((v55 | v5) & 0x4000000000000000LL) != 0 )
    goto LABEL_26;
  if ( v5 != v55 )
    goto LABEL_26;
  v49 = v5 & 0x3FFFFFFFFFFFFFFFLL;
  if ( 1LL << v12 <= (v5 & 0x3FFFFFFFFFFFFFFFLL) || v14 % v49 || v52 % (v55 & 0x3FFFFFFFFFFFFFFFLL) )
    goto LABEL_26;
  v50 = v14 & ~(-1LL << v12);
  v51 = v52 & ~(-1LL << v13);
  if ( (__int64)(v50 + v49) <= v51 )
    return 0;
  if ( (__int64)((v55 & 0x3FFFFFFFFFFFFFFFLL) + v51) <= v50 )
    return 0;
LABEL_26:
  v53 = v55;
  v56 = (__int64 *)v58;
  v59 = sub_C52410();
  v15 = sub_C959E0();
  v16 = (char *)v59[2];
  v17 = (char *)(v59 + 1);
  if ( v16 )
  {
    v18 = (char *)(v59 + 1);
    do
    {
      while ( 1 )
      {
        v19 = *((_QWORD *)v16 + 2);
        v20 = *((_QWORD *)v16 + 3);
        if ( v15 <= *((_QWORD *)v16 + 4) )
          break;
        v16 = (char *)*((_QWORD *)v16 + 3);
        if ( !v20 )
          goto LABEL_31;
      }
      v18 = v16;
      v16 = (char *)*((_QWORD *)v16 + 2);
    }
    while ( v19 );
LABEL_31:
    if ( v17 != v18 && v15 >= *((_QWORD *)v18 + 4) )
      v17 = v18;
  }
  v21 = (char *)sub_C52410();
  v26 = v56;
  v27 = v53;
  if ( v17 == v21 + 8 )
    goto LABEL_78;
  v28 = *((_QWORD *)v17 + 7);
  v25 = v17 + 48;
  if ( !v28 )
    goto LABEL_78;
  v15 = (unsigned int)dword_5038468;
  v29 = v17 + 48;
  do
  {
    while ( 1 )
    {
      v23 = *(_QWORD *)(v28 + 16);
      v22 = *(_QWORD *)(v28 + 24);
      if ( *(_DWORD *)(v28 + 32) >= dword_5038468 )
        break;
      v28 = *(_QWORD *)(v28 + 24);
      if ( !v22 )
        goto LABEL_40;
    }
    v29 = (_QWORD *)v28;
    v28 = *(_QWORD *)(v28 + 16);
  }
  while ( v23 );
LABEL_40:
  if ( v29 == v25
    || dword_5038468 < *((_DWORD *)v29 + 8)
    || (v22 = *((unsigned int *)v29 + 9), v30 = qword_50384E8, (int)v22 <= 0) )
  {
LABEL_78:
    v48 = *(_QWORD *)(*(_QWORD *)(*a1 + 40LL) + 16LL);
    v30 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, __int64, __int64, __int64, _QWORD *))(*(_QWORD *)v48 + 408LL))(
            v48,
            v15,
            v22,
            v23,
            v24,
            v25);
    v27 = v53;
    v26 = v56;
  }
  if ( !v30 )
    return 1;
  v31 = (__int64 *)a1[113];
  if ( !v31 )
    return 1;
  v32 = *v6;
  if ( !*v6 )
    return 1;
  if ( (v32 & 4) != 0 )
    return 1;
  if ( (v32 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    return 1;
  v33 = *v26;
  if ( !*v26 )
    return 1;
  if ( (v33 & 4) != 0 )
    return 1;
  v34 = v33 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v34 || !v9 || !v54 || (v5 & 0x4000000000000000LL) != 0 && v14 )
    return 1;
  v35 = v27 & 0x4000000000000000LL;
  if ( v52 )
  {
    if ( v35 )
      return 1;
  }
  v36 = v52;
  if ( v52 > v14 )
    v36 = v14;
  if ( (v5 & 0x4000000000000000LL) == 0 )
  {
    v5 = (v5 & 0x3FFFFFFFFFFFFFFFLL) + v14 - v36;
    if ( v5 > 0x3FFFFFFFFFFFFFFBLL )
      v5 = 0xBFFFFFFFFFFFFFFELL;
  }
  if ( !v35 )
  {
    v27 = v52 + (v27 & 0x3FFFFFFFFFFFFFFFLL) - v36;
    if ( v27 > 0x3FFFFFFFFFFFFFFBLL )
      v27 = 0xBFFFFFFFFFFFFFFELL;
  }
  if ( (_BYTE)qword_5038408 )
  {
    v37 = v26[5];
    v38 = v26[6];
    v75 = v34;
    v39 = v26[7];
    v40 = v26[8];
    v76 = v27;
    v77 = v37;
    v41 = v6[8];
    v78 = v38;
    v42 = v6[7];
    v79 = v39;
    v43 = v6[6];
    v80 = v40;
    v44 = v6[5];
  }
  else
  {
    v75 = v34;
    v41 = 0;
    v42 = 0;
    v43 = 0;
    v76 = v27;
    v44 = 0;
    v77 = 0;
    v78 = 0;
    v79 = 0;
    v80 = 0;
  }
  v45 = *v6;
  if ( !*v6 || (v45 & 4) != 0 )
    v46 = 0;
  else
    v46 = v45 & 0xFFFFFFFFFFFFFFF8LL;
  v74[2] = v44;
  v74[3] = v43;
  v74[4] = v42;
  v74[5] = v41;
  v74[0] = v46;
  v74[1] = v5;
  return (unsigned __int8)sub_CF4D50(*v31, (__int64)v74, (__int64)&v75, (__int64)(v31 + 1), 0) != 0;
}
