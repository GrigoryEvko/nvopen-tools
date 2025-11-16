// Function: sub_20A1C00
// Address: 0x20a1c00
//
__int64 __fastcall sub_20A1C00(__int64 a1, _QWORD *a2, int a3)
{
  unsigned __int64 v5; // r14
  _QWORD *v6; // rdi
  __int64 v7; // rax
  char v8; // al
  unsigned __int64 v9; // r14
  _QWORD *v10; // rdi
  __int64 v11; // rax
  char v12; // al
  unsigned __int64 v13; // r14
  _QWORD *v14; // rdi
  __int64 v15; // rax
  char v16; // al
  unsigned __int64 v17; // r14
  _QWORD *v18; // rdi
  __int64 v19; // rax
  char v20; // al
  unsigned __int64 v21; // r14
  _QWORD *v22; // rdi
  __int64 v23; // rax
  char v24; // al
  unsigned __int64 v25; // r14
  _QWORD *v26; // rdi
  __int64 v27; // rax
  char v28; // al
  unsigned __int64 v29; // r14
  _QWORD *v30; // rdi
  __int64 v31; // rax
  char v32; // al
  unsigned __int64 v33; // r14
  _QWORD *v34; // rdi
  __int64 v35; // rax
  char v36; // al
  unsigned __int64 v37; // r14
  _QWORD *v38; // rdi
  __int64 v39; // rax
  char v40; // al
  unsigned __int64 v41; // r14
  _QWORD *v42; // rdi
  __int64 v43; // rax
  char v44; // al
  __int64 result; // rax
  _QWORD v46[5]; // [rsp+8h] [rbp-28h] BYREF

  v5 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  v6 = (_QWORD *)(v5 + 56);
  if ( (*a2 & 4) != 0 )
  {
    if ( !(unsigned __int8)sub_1560290(v6, a3, 40) )
    {
      v7 = *(_QWORD *)(v5 - 24);
      if ( *(_BYTE *)(v7 + 16) )
      {
LABEL_4:
        v8 = 0;
        goto LABEL_8;
      }
      goto LABEL_7;
    }
  }
  else if ( !(unsigned __int8)sub_1560290(v6, a3, 40) )
  {
    v7 = *(_QWORD *)(v5 - 72);
    if ( *(_BYTE *)(v7 + 16) )
      goto LABEL_4;
LABEL_7:
    v46[0] = *(_QWORD *)(v7 + 112);
    v8 = sub_1560290(v46, a3, 40);
    goto LABEL_8;
  }
  v8 = 1;
LABEL_8:
  *(_BYTE *)(a1 + 32) = v8 & 1 | *(_BYTE *)(a1 + 32) & 0xFE;
  v9 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  v10 = (_QWORD *)(v9 + 56);
  if ( (*a2 & 4) != 0 )
  {
    if ( !(unsigned __int8)sub_1560290(v10, a3, 58) )
    {
      v11 = *(_QWORD *)(v9 - 24);
      if ( !*(_BYTE *)(v11 + 16) )
      {
LABEL_11:
        v46[0] = *(_QWORD *)(v11 + 112);
        v12 = sub_1560290(v46, a3, 58);
        goto LABEL_12;
      }
      goto LABEL_72;
    }
  }
  else if ( !(unsigned __int8)sub_1560290(v10, a3, 58) )
  {
    v11 = *(_QWORD *)(v9 - 72);
    if ( !*(_BYTE *)(v11 + 16) )
      goto LABEL_11;
LABEL_72:
    v12 = 0;
    goto LABEL_12;
  }
  v12 = 1;
LABEL_12:
  *(_BYTE *)(a1 + 32) = *(_BYTE *)(a1 + 32) & 0xFD | (2 * (v12 & 1));
  v13 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  v14 = (_QWORD *)(v13 + 56);
  if ( (*a2 & 4) != 0 )
  {
    if ( !(unsigned __int8)sub_1560290(v14, a3, 12) )
    {
      v15 = *(_QWORD *)(v13 - 24);
      if ( !*(_BYTE *)(v15 + 16) )
      {
LABEL_15:
        v46[0] = *(_QWORD *)(v15 + 112);
        v16 = sub_1560290(v46, a3, 12);
        goto LABEL_16;
      }
      goto LABEL_69;
    }
  }
  else if ( !(unsigned __int8)sub_1560290(v14, a3, 12) )
  {
    v15 = *(_QWORD *)(v13 - 72);
    if ( !*(_BYTE *)(v15 + 16) )
      goto LABEL_15;
LABEL_69:
    v16 = 0;
    goto LABEL_16;
  }
  v16 = 1;
LABEL_16:
  *(_BYTE *)(a1 + 32) = *(_BYTE *)(a1 + 32) & 0xFB | (4 * (v16 & 1));
  v17 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  v18 = (_QWORD *)(v17 + 56);
  if ( (*a2 & 4) != 0 )
  {
    if ( !(unsigned __int8)sub_1560290(v18, a3, 53) )
    {
      v19 = *(_QWORD *)(v17 - 24);
      if ( !*(_BYTE *)(v19 + 16) )
      {
LABEL_19:
        v46[0] = *(_QWORD *)(v19 + 112);
        v20 = sub_1560290(v46, a3, 53);
        goto LABEL_20;
      }
      goto LABEL_66;
    }
  }
  else if ( !(unsigned __int8)sub_1560290(v18, a3, 53) )
  {
    v19 = *(_QWORD *)(v17 - 72);
    if ( !*(_BYTE *)(v19 + 16) )
      goto LABEL_19;
LABEL_66:
    v20 = 0;
    goto LABEL_20;
  }
  v20 = 1;
LABEL_20:
  *(_BYTE *)(a1 + 32) = *(_BYTE *)(a1 + 32) & 0xF7 | (8 * (v20 & 1));
  v21 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  v22 = (_QWORD *)(v21 + 56);
  if ( (*a2 & 4) != 0 )
  {
    if ( !(unsigned __int8)sub_1560290(v22, a3, 19) )
    {
      v23 = *(_QWORD *)(v21 - 24);
      if ( !*(_BYTE *)(v23 + 16) )
      {
LABEL_23:
        v46[0] = *(_QWORD *)(v23 + 112);
        v24 = sub_1560290(v46, a3, 19);
        goto LABEL_24;
      }
      goto LABEL_63;
    }
  }
  else if ( !(unsigned __int8)sub_1560290(v22, a3, 19) )
  {
    v23 = *(_QWORD *)(v21 - 72);
    if ( !*(_BYTE *)(v23 + 16) )
      goto LABEL_23;
LABEL_63:
    v24 = 0;
    goto LABEL_24;
  }
  v24 = 1;
LABEL_24:
  *(_BYTE *)(a1 + 32) = *(_BYTE *)(a1 + 32) & 0xEF | (16 * (v24 & 1));
  v25 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  v26 = (_QWORD *)(v25 + 56);
  if ( (*a2 & 4) != 0 )
  {
    if ( !(unsigned __int8)sub_1560290(v26, a3, 6) )
    {
      v27 = *(_QWORD *)(v25 - 24);
      if ( !*(_BYTE *)(v27 + 16) )
      {
LABEL_27:
        v46[0] = *(_QWORD *)(v27 + 112);
        v28 = sub_1560290(v46, a3, 6);
        goto LABEL_28;
      }
      goto LABEL_60;
    }
  }
  else if ( !(unsigned __int8)sub_1560290(v26, a3, 6) )
  {
    v27 = *(_QWORD *)(v25 - 72);
    if ( !*(_BYTE *)(v27 + 16) )
      goto LABEL_27;
LABEL_60:
    v28 = 0;
    goto LABEL_28;
  }
  v28 = 1;
LABEL_28:
  *(_BYTE *)(a1 + 32) = *(_BYTE *)(a1 + 32) & 0xDF | (32 * (v28 & 1));
  v29 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  v30 = (_QWORD *)(v29 + 56);
  if ( (*a2 & 4) != 0 )
  {
    if ( !(unsigned __int8)sub_1560290(v30, a3, 11) )
    {
      v31 = *(_QWORD *)(v29 - 24);
      if ( !*(_BYTE *)(v31 + 16) )
      {
LABEL_31:
        v46[0] = *(_QWORD *)(v31 + 112);
        v32 = sub_1560290(v46, a3, 11);
        goto LABEL_32;
      }
      goto LABEL_57;
    }
  }
  else if ( !(unsigned __int8)sub_1560290(v30, a3, 11) )
  {
    v31 = *(_QWORD *)(v29 - 72);
    if ( !*(_BYTE *)(v31 + 16) )
      goto LABEL_31;
LABEL_57:
    v32 = 0;
    goto LABEL_32;
  }
  v32 = 1;
LABEL_32:
  *(_BYTE *)(a1 + 32) = *(_BYTE *)(a1 + 32) & 0xBF | ((v32 & 1) << 6);
  v33 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  v34 = (_QWORD *)(v33 + 56);
  if ( (*a2 & 4) != 0 )
  {
    if ( !(unsigned __int8)sub_1560290(v34, a3, 38) )
    {
      v35 = *(_QWORD *)(v33 - 24);
      if ( !*(_BYTE *)(v35 + 16) )
      {
LABEL_35:
        v46[0] = *(_QWORD *)(v35 + 112);
        v36 = sub_1560290(v46, a3, 38);
        goto LABEL_36;
      }
      goto LABEL_54;
    }
  }
  else if ( !(unsigned __int8)sub_1560290(v34, a3, 38) )
  {
    v35 = *(_QWORD *)(v33 - 72);
    if ( !*(_BYTE *)(v35 + 16) )
      goto LABEL_35;
LABEL_54:
    v36 = 0;
    goto LABEL_36;
  }
  v36 = 1;
LABEL_36:
  *(_BYTE *)(a1 + 32) = (v36 << 7) | *(_BYTE *)(a1 + 32) & 0x7F;
  v37 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  v38 = (_QWORD *)(v37 + 56);
  if ( (*a2 & 4) != 0 )
  {
    if ( !(unsigned __int8)sub_1560290(v38, a3, 55) )
    {
      v39 = *(_QWORD *)(v37 - 24);
      if ( !*(_BYTE *)(v39 + 16) )
      {
LABEL_39:
        v46[0] = *(_QWORD *)(v39 + 112);
        v40 = sub_1560290(v46, a3, 55);
        goto LABEL_40;
      }
      goto LABEL_51;
    }
  }
  else if ( !(unsigned __int8)sub_1560290(v38, a3, 55) )
  {
    v39 = *(_QWORD *)(v37 - 72);
    if ( !*(_BYTE *)(v39 + 16) )
      goto LABEL_39;
LABEL_51:
    v40 = 0;
    goto LABEL_40;
  }
  v40 = 1;
LABEL_40:
  *(_BYTE *)(a1 + 33) = v40 & 1 | *(_BYTE *)(a1 + 33) & 0xFE;
  v41 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  v42 = (_QWORD *)(v41 + 56);
  if ( (*a2 & 4) == 0 )
  {
    if ( !(unsigned __int8)sub_1560290(v42, a3, 54) )
    {
      v43 = *(_QWORD *)(v41 - 72);
      if ( !*(_BYTE *)(v43 + 16) )
        goto LABEL_43;
LABEL_48:
      v44 = 0;
      goto LABEL_44;
    }
LABEL_78:
    v44 = 1;
    goto LABEL_44;
  }
  if ( (unsigned __int8)sub_1560290(v42, a3, 54) )
    goto LABEL_78;
  v43 = *(_QWORD *)(v41 - 24);
  if ( *(_BYTE *)(v43 + 16) )
    goto LABEL_48;
LABEL_43:
  v46[0] = *(_QWORD *)(v43 + 112);
  v44 = sub_1560290(v46, a3, 54);
LABEL_44:
  *(_BYTE *)(a1 + 33) = *(_BYTE *)(a1 + 33) & 0xFD | (2 * (v44 & 1));
  result = sub_15603A0((_QWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 56), a3);
  *(_WORD *)(a1 + 34) = result;
  return result;
}
