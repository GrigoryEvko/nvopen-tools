// Function: sub_340F940
// Address: 0x340f940
//
unsigned __int8 *__fastcall sub_340F940(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  __int64 v8; // r10
  signed int v9; // r12d
  __int64 v10; // rax
  __int64 v11; // r11
  __int16 v12; // bx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v16; // rsi
  bool v17; // al
  __int64 v18; // rcx
  bool v19; // al
  __int64 v20; // r14
  void *v21; // rax
  __int64 v22; // rax
  int v23; // r12d
  __int64 v24; // r13
  __int64 v25; // rbx
  int v26; // eax
  char v27; // al
  __int64 v28; // r9
  __int64 v29; // rbx
  void *v30; // rax
  __int64 v31; // rdi
  unsigned int v32; // eax
  char v33; // si
  bool v34; // zf
  char v35; // al
  signed int v36; // eax
  unsigned int v37; // ebx
  __int64 v38; // r14
  __int128 v39; // rax
  __int64 v40; // r9
  __int128 v41; // [rsp-20h] [rbp-B0h]
  __int64 v42; // [rsp+0h] [rbp-90h]
  __int64 v44; // [rsp+8h] [rbp-88h]
  __int64 v45; // [rsp+8h] [rbp-88h]
  _QWORD *v46; // [rsp+8h] [rbp-88h]
  __int64 v47; // [rsp+10h] [rbp-80h]
  int v48; // [rsp+18h] [rbp-78h]
  __int64 v49; // [rsp+18h] [rbp-78h]
  __int64 v50; // [rsp+18h] [rbp-78h]
  __int64 v51; // [rsp+18h] [rbp-78h]
  int v52; // [rsp+18h] [rbp-78h]
  __int64 v53; // [rsp+18h] [rbp-78h]
  __int64 v54; // [rsp+20h] [rbp-70h] BYREF
  __int64 v55; // [rsp+28h] [rbp-68h]
  __int128 v56; // [rsp+30h] [rbp-60h] BYREF
  __int64 v57[10]; // [rsp+40h] [rbp-50h] BYREF
  __int64 v58; // [rsp+B0h] [rbp+20h]
  __int64 v59; // [rsp+B0h] [rbp+20h]
  __int64 v60; // [rsp+B0h] [rbp+20h]
  __int64 v61; // [rsp+B0h] [rbp+20h]
  __int64 v62; // [rsp+B0h] [rbp+20h]

  v8 = a1;
  v9 = a6;
  v10 = *(_QWORD *)(a4 + 48) + 16LL * (unsigned int)a5;
  v11 = a8;
  v54 = a2;
  v55 = a3;
  v12 = *(_WORD *)v10;
  v13 = *(_QWORD *)(v10 + 8);
  v57[1] = a1;
  v57[2] = (__int64)&v56;
  *((_QWORD *)&v56 + 1) = v13;
  LOWORD(v56) = v12;
  v57[0] = (__int64)&v54;
  v57[3] = a8;
  if ( (_DWORD)a6 == 16 )
    return sub_3401740(v8, 0, v11, (unsigned int)v54, v55, a6, v56);
  v14 = a4;
  if ( (unsigned int)a6 > 0x10 )
  {
    if ( (_DWORD)a6 == 23 )
      return sub_3401740(v8, 1, v11, (unsigned int)v54, v55, a6, v56);
  }
  else
  {
    if ( !(_DWORD)a6 )
      return sub_3401740(v8, 0, v11, (unsigned int)v54, v55, a6, v56);
    if ( (_DWORD)a6 == 15 )
      return sub_3401740(v8, 1, v11, (unsigned int)v54, v55, a6, v56);
  }
  v16 = *(unsigned int *)(a4 + 24);
  if ( v12 )
  {
    if ( (unsigned __int16)(v12 - 2) > 7u
      && (unsigned __int16)(v12 - 17) > 0x6Cu
      && (unsigned __int16)(v12 - 176) > 0x1Fu )
    {
      goto LABEL_9;
    }
LABEL_37:
    if ( (_DWORD)v16 == 51 )
    {
      if ( v9 != 17 && v9 != 22 && *(_DWORD *)(a7 + 24) != 51 )
        goto LABEL_52;
    }
    else
    {
      v18 = *(unsigned int *)(a7 + 24);
      if ( (_DWORD)v18 != 51 )
      {
        if ( v14 != (_QWORD)a7 || (_DWORD)a5 != DWORD2(a7) )
          goto LABEL_10;
LABEL_52:
        v33 = v9 & 1;
        return sub_3401740(v8, v33, v11, (unsigned int)v54, v55, a6, v56);
      }
      if ( v9 != 17 && v9 != 22 )
        goto LABEL_52;
    }
    return sub_3400FF0(v57, v16);
  }
  v42 = a5;
  v17 = sub_3007070((__int64)&v56);
  v16 = (unsigned int)v16;
  v8 = a1;
  v14 = a4;
  a5 = v42;
  v11 = a8;
  if ( v17 )
    goto LABEL_37;
LABEL_9:
  v18 = *(unsigned int *)(a7 + 24);
LABEL_10:
  if ( ((_DWORD)v18 == 11 || (_DWORD)v18 == 35) && ((_DWORD)v16 == 35 || (_DWORD)v16 == 11) )
  {
    v24 = *(_QWORD *)(v14 + 96);
    v25 = *(_QWORD *)(a7 + 96);
    v59 = v11;
    v50 = v8;
    v26 = sub_34B9240((unsigned int)v9, v16, v14, v18, a5);
    v27 = sub_B532C0(v24 + 24, (_QWORD *)(v25 + 24), v26);
    return sub_3401740(v50, v27, v59, (unsigned int)v54, v55, v28, v56);
  }
  v19 = (_DWORD)v18 == 12 || (_DWORD)v18 == 36;
  if ( (_DWORD)v16 != 36 && (_DWORD)v16 != 12 )
  {
    if ( v19 )
    {
      v20 = *(_QWORD *)(a7 + 96);
      v58 = v11;
      v44 = v8;
      v48 = v18;
      v21 = sub_C33340();
      LODWORD(v18) = v48;
      v16 = (unsigned int)v16;
      v34 = *(_QWORD *)(v20 + 24) == (_QWORD)v21;
      v11 = v58;
      v22 = v20 + 24;
      v8 = v44;
      if ( v34 )
        v22 = *(_QWORD *)(v20 + 32);
      if ( (*(_BYTE *)(v22 + 20) & 7) == 1 )
        goto LABEL_30;
    }
    if ( v12 )
    {
      if ( (unsigned __int16)(v12 - 10) > 6u
        && (unsigned __int16)(v12 - 126) > 0x31u
        && (unsigned __int16)(v12 - 208) > 0x14u )
      {
        return 0;
      }
LABEL_22:
      if ( (_DWORD)v16 != 51 && (_DWORD)v18 != 51 )
        return 0;
LABEL_30:
      v23 = (v9 >> 3) & 3;
      if ( v23 == 1 )
        return sub_3401740(v8, 1, v11, (unsigned int)v54, v55, a6, v56);
      if ( v23 != 2 )
      {
        if ( v23 )
          BUG();
        return sub_3401740(v8, 0, v11, (unsigned int)v54, v55, a6, v56);
      }
      return sub_3400FF0(v57, v16);
    }
    goto LABEL_82;
  }
  if ( v19 )
  {
    v29 = *(_QWORD *)(v14 + 96);
    v60 = v11;
    v51 = v8;
    v30 = sub_C33340();
    v31 = v29 + 24;
    v16 = *(_QWORD *)(a7 + 96) + 24LL;
    if ( *(void **)(v29 + 24) == v30 )
    {
      v32 = sub_C3E510(v31, v16);
      v11 = v60;
      v8 = v51;
    }
    else
    {
      v32 = sub_C37950(v31, v16);
      v8 = v51;
      v11 = v60;
    }
    switch ( v9 )
    {
      case 1:
        goto LABEL_59;
      case 2:
        goto LABEL_71;
      case 3:
        goto LABEL_69;
      case 4:
        goto LABEL_65;
      case 5:
        goto LABEL_63;
      case 6:
        goto LABEL_67;
      case 7:
        v33 = v32 != 3;
        return sub_3401740(v8, v33, v11, (unsigned int)v54, v55, a6, v56);
      case 8:
        v34 = v32 == 3;
        goto LABEL_60;
      case 9:
        v32 &= ~2u;
        goto LABEL_59;
      case 10:
        v32 -= 2;
        goto LABEL_63;
      case 11:
        v33 = v32 != 0;
        return sub_3401740(v8, v33, v11, (unsigned int)v54, v55, a6, v56);
      case 12:
        v33 = v32 == 0 || v32 == 3;
        return sub_3401740(v8, v33, v11, (unsigned int)v54, v55, a6, v56);
      case 13:
        v33 = v32 != 2;
        return sub_3401740(v8, v33, v11, (unsigned int)v54, v55, a6, v56);
      case 14:
        v33 = v32 != 1;
        return sub_3401740(v8, v33, v11, (unsigned int)v54, v55, a6, v56);
      case 17:
        if ( v32 == 3 )
          return sub_3400FF0(v57, v16);
LABEL_59:
        v34 = v32 == 1;
        goto LABEL_60;
      case 18:
        if ( v32 == 3 )
          return sub_3400FF0(v57, v16);
LABEL_71:
        v34 = v32 == 2;
        goto LABEL_60;
      case 19:
        if ( v32 == 3 )
          return sub_3400FF0(v57, v16);
LABEL_69:
        v33 = v32 - 1 <= 1;
        return sub_3401740(v8, v33, v11, (unsigned int)v54, v55, a6, v56);
      case 20:
        if ( v32 == 3 )
          return sub_3400FF0(v57, v16);
LABEL_65:
        v34 = v32 == 0;
        goto LABEL_60;
      case 21:
        if ( v32 == 3 )
          return sub_3400FF0(v57, v16);
LABEL_63:
        v33 = v32 <= 1;
        return sub_3401740(v8, v33, v11, (unsigned int)v54, v55, a6, v56);
      case 22:
        if ( v32 == 3 )
          return sub_3400FF0(v57, v16);
LABEL_67:
        v34 = (v32 & 0xFFFFFFFD) == 0;
LABEL_60:
        v33 = v34;
        break;
      default:
        return 0;
    }
    return sub_3401740(v8, v33, v11, (unsigned int)v54, v55, a6, v56);
  }
  v47 = a5;
  v49 = v14;
  if ( !v12 )
  {
LABEL_82:
    v61 = v11;
    v45 = v8;
    v52 = v18;
    v35 = sub_3007030((__int64)&v56);
    LODWORD(v18) = v52;
    v16 = (unsigned int)v16;
    v8 = v45;
    v11 = v61;
    if ( v35 )
      goto LABEL_22;
    return 0;
  }
  if ( (_DWORD)v18 == 51 )
  {
    if ( (unsigned __int16)(v12 - 10) > 6u
      && (unsigned __int16)(v12 - 126) > 0x31u
      && (unsigned __int16)(v12 - 208) > 0x14u )
    {
      return 0;
    }
    goto LABEL_30;
  }
  v62 = v11;
  v46 = (_QWORD *)v8;
  v36 = sub_33CBD20(v9);
  if ( ((*(_DWORD *)(v46[2] + 4 * (((unsigned __int16)v56 >> 3) + 36LL * v36 - v36) + 521536) >> (4 * (v56 & 7))) & 0xF) != 0 )
    return 0;
  v37 = v54;
  v38 = v49;
  v53 = v55;
  *(_QWORD *)&v39 = sub_33ED040(v46, v36);
  *((_QWORD *)&v41 + 1) = v47;
  *(_QWORD *)&v41 = v38;
  return (unsigned __int8 *)sub_340F900(v46, 0xD0u, v62, v37, v53, v40, a7, v41, v39);
}
