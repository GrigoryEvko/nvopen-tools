// Function: sub_172AD80
// Address: 0x172ad80
//
unsigned __int8 *__fastcall sub_172AD80(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        char a4,
        __int64 a5,
        double a6,
        double a7,
        double a8)
{
  __int64 v10; // rdi
  __int64 v11; // r13
  unsigned int v12; // ecx
  __int64 v15; // rax
  char v16; // dl
  __int64 v17; // r12
  __int64 v19; // r14
  unsigned __int8 *v20; // rcx
  __int64 v21; // rax
  char v22; // dl
  __int64 v23; // r15
  unsigned __int8 *v24; // r13
  __int64 v25; // rdi
  unsigned __int8 *v26; // rax
  __int64 v27; // rdi
  __int64 v28; // r12
  unsigned __int8 *v29; // rax
  __int64 v30; // r13
  __int64 v31; // rsi
  __int16 v32; // ax
  __int64 v33; // rax
  int v35; // [rsp+8h] [rbp-58h]
  __int64 v36[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v37; // [rsp+20h] [rbp-40h]

  v10 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v10 + 16) != 13 )
  {
    if ( *(_QWORD *)(a3 - 24) )
      return 0;
LABEL_35:
    BUG();
  }
  v11 = *(_QWORD *)(a3 - 24);
  if ( !v11 )
    goto LABEL_35;
  if ( *(_BYTE *)(v11 + 16) != 13 )
    return 0;
  v12 = *(_DWORD *)(v10 + 32);
  if ( !(v12 <= 0x40 ? *(_QWORD *)(v10 + 24) == 0 : v12 == (unsigned int)sub_16A57B0(v10 + 24)) )
    return 0;
  if ( *(_DWORD *)(v11 + 32) <= 0x40u )
  {
    if ( *(_QWORD *)(v11 + 24) )
      return 0;
    v15 = *(_QWORD *)(a2 - 48);
    v16 = *(_BYTE *)(v15 + 16);
    if ( v16 != 50 )
      goto LABEL_10;
LABEL_16:
    v19 = *(_QWORD *)(v15 - 48);
    if ( !v19 )
      return 0;
    v20 = *(unsigned __int8 **)(v15 - 24);
    if ( !v20 )
      return 0;
    v21 = *(_QWORD *)(a3 - 48);
    v22 = *(_BYTE *)(v21 + 16);
    if ( v22 != 50 )
      goto LABEL_40;
LABEL_19:
    v23 = *(_QWORD *)(v21 - 48);
    if ( !v23 )
      return 0;
    v24 = *(unsigned __int8 **)(v21 - 24);
    if ( !v24 )
      return 0;
    goto LABEL_21;
  }
  v35 = *(_DWORD *)(v11 + 32);
  if ( v35 != (unsigned int)sub_16A57B0(v11 + 24) )
    return 0;
  v15 = *(_QWORD *)(a2 - 48);
  v16 = *(_BYTE *)(v15 + 16);
  if ( v16 == 50 )
    goto LABEL_16;
LABEL_10:
  if ( v16 != 5 )
    return 0;
  if ( *(_WORD *)(v15 + 18) != 26 )
    return 0;
  v19 = *(_QWORD *)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF));
  if ( !v19 )
    return 0;
  v20 = *(unsigned __int8 **)(v15 + 24 * (1LL - (*(_DWORD *)(v15 + 20) & 0xFFFFFFF)));
  if ( !v20 )
    return 0;
  v21 = *(_QWORD *)(a3 - 48);
  v22 = *(_BYTE *)(v21 + 16);
  if ( v22 == 50 )
    goto LABEL_19;
LABEL_40:
  if ( v22 != 5 )
    return 0;
  if ( *(_WORD *)(v21 + 18) != 26 )
    return 0;
  v23 = *(_QWORD *)(v21 - 24LL * (*(_DWORD *)(v21 + 20) & 0xFFFFFFF));
  if ( !v23 )
    return 0;
  v24 = *(unsigned __int8 **)(v21 + 24 * (1LL - (*(_DWORD *)(v21 + 20) & 0xFFFFFFF)));
  if ( !v24 )
    return 0;
LABEL_21:
  if ( (unsigned __int8 *)v19 == v24 )
  {
    if ( v20 == (unsigned __int8 *)v19 )
      goto LABEL_27;
    goto LABEL_26;
  }
  if ( v20 != v24 )
  {
    if ( v20 == (unsigned __int8 *)v23 )
    {
      v23 = (__int64)v24;
      v24 = v20;
      goto LABEL_27;
    }
    if ( v23 == v19 )
    {
      v23 = (__int64)v24;
      v24 = (unsigned __int8 *)v19;
LABEL_26:
      v19 = (__int64)v20;
      goto LABEL_27;
    }
    return 0;
  }
LABEL_27:
  if ( !(unsigned __int8)sub_14BDFF0(v19, a1[333], 0, 0, a1[330], a5, a1[332])
    || !(unsigned __int8)sub_14BDFF0(v23, a1[333], 0, 0, a1[330], a5, a1[332]) )
  {
    return 0;
  }
  v25 = a1[1];
  v37 = 257;
  v26 = sub_172AC10(v25, v19, v23, v36, a6, a7, a8);
  v27 = a1[1];
  v28 = (__int64)v26;
  v37 = 257;
  v29 = sub_1729500(v27, v24, (__int64)v26, v36, a6, a7, a8);
  v30 = a1[1];
  v31 = (__int64)v29;
  v37 = 257;
  v32 = (a4 == 0) + 32;
  if ( *(_BYTE *)(v31 + 16) > 0x10u || *(_BYTE *)(v28 + 16) > 0x10u )
    return sub_1727440(v30, v32, v31, v28, v36);
  v17 = sub_15A37B0(v32, (_QWORD *)v31, (_QWORD *)v28, 0);
  v33 = sub_14DBA30(v17, *(_QWORD *)(v30 + 96), 0);
  if ( v33 )
    return (unsigned __int8 *)v33;
  return (unsigned __int8 *)v17;
}
