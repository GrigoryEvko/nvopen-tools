// Function: sub_5C8A10
// Address: 0x5c8a10
//
_QWORD *__fastcall sub_5C8A10(__int64 a1, _QWORD *a2, char a3)
{
  _QWORD *v3; // r12
  __int64 v4; // rbx
  _QWORD *v5; // r15
  __int64 v7; // rdx
  _QWORD *v8; // r10
  _BYTE *v9; // r14
  __int64 v10; // rax
  __int64 v11; // rdx
  _QWORD *v12; // r10
  char *v14; // rax
  _QWORD *v15; // r10
  __int64 v16; // rax
  bool v17; // al
  __int64 v18; // rax
  __int64 v19; // rax
  _QWORD *v20; // r14
  _QWORD *v21; // r15
  __int64 v22; // rbx
  _QWORD *v23; // r12
  __int64 v24; // rax
  size_t v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  _QWORD *v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rax
  const char *v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // [rsp+0h] [rbp-80h]
  __int64 v38; // [rsp+10h] [rbp-70h]
  _QWORD *v39; // [rsp+18h] [rbp-68h]
  _QWORD *v40; // [rsp+20h] [rbp-60h]
  _QWORD *v41; // [rsp+20h] [rbp-60h]
  _QWORD *v42; // [rsp+20h] [rbp-60h]
  _QWORD *v43; // [rsp+20h] [rbp-60h]
  __int64 v44; // [rsp+20h] [rbp-60h]
  __int64 v45; // [rsp+28h] [rbp-58h]
  _QWORD *v46; // [rsp+28h] [rbp-58h]
  __int64 v47; // [rsp+28h] [rbp-58h]
  __int64 v48; // [rsp+28h] [rbp-58h]
  _QWORD *v49; // [rsp+28h] [rbp-58h]
  _BYTE *v50; // [rsp+28h] [rbp-58h]
  __int64 v51; // [rsp+28h] [rbp-58h]
  __int64 v52; // [rsp+28h] [rbp-58h]
  char *v53; // [rsp+28h] [rbp-58h]
  __int64 v54; // [rsp+30h] [rbp-50h]
  _QWORD *v55; // [rsp+38h] [rbp-48h]
  __int64 v56; // [rsp+38h] [rbp-48h]
  __int64 v57; // [rsp+38h] [rbp-48h]
  __int64 v58; // [rsp+38h] [rbp-48h]
  __int64 v59; // [rsp+38h] [rbp-48h]
  size_t v60; // [rsp+38h] [rbp-48h]
  __int64 v61; // [rsp+38h] [rbp-48h]
  unsigned int v62; // [rsp+44h] [rbp-3Ch] BYREF
  __int64 v63[7]; // [rsp+48h] [rbp-38h] BYREF

  v3 = a2;
  v4 = a1;
  if ( unk_4F077C4 != 2 )
  {
    sub_684B30(2646, a1 + 56);
    *(_BYTE *)(a1 + 8) = 0;
    return v3;
  }
  v5 = *(_QWORD **)(a1 + 32);
  if ( !v5 )
  {
    if ( a3 != 28 )
    {
      v14 = sub_5C79F0(a1);
      sub_6851A0(1833, a1 + 56, v14);
      *(_BYTE *)(a1 + 8) = 0;
      return v3;
    }
    v7 = *(_QWORD *)(a1 + 48);
    goto LABEL_39;
  }
  v7 = *(_QWORD *)(a1 + 48);
  switch ( a3 )
  {
    case 11:
      if ( unk_4F077A8 <= 0x9FC3u )
      {
        if ( (*((_BYTE *)a2 + 195) & 1) == 0 )
        {
          v9 = a2;
          v8 = 0;
          goto LABEL_12;
        }
LABEL_37:
        *(_BYTE *)(v4 + 8) = 0;
        return v3;
      }
      v55 = 0;
      v9 = a2;
      v17 = v7 != 0;
      v5 = 0;
      goto LABEL_52;
    case 7:
      v55 = 0;
      v5 = a2;
      goto LABEL_27;
    case 28:
LABEL_39:
      if ( unk_4F077A8 <= 0xC34Fu )
      {
        v56 = v7;
        sub_684B30(1865, a1 + 56);
        *(_BYTE *)(a1 + 8) = 0;
        v7 = v56;
      }
      else if ( (*((_BYTE *)a2 + 124) & 2) != 0 )
      {
        if ( a2[1] )
        {
          if ( v5 )
          {
LABEL_43:
            v55 = a2;
            v5 = 0;
LABEL_44:
            v17 = v7 != 0;
            if ( a3 == 6 && v7 )
            {
              v15 = 0;
              goto LABEL_47;
            }
            v9 = 0;
LABEL_52:
            if ( a3 != 11 || !v17 )
            {
              v15 = 0;
LABEL_29:
              v41 = v15;
              v47 = v7;
              if ( *(_BYTE *)(v4 + 8) != 80 )
                return v3;
              v16 = sub_736C60(80, a2[13]);
              v11 = v47;
              v12 = v41;
              v54 = v16;
              if ( a3 == 28 && v4 != v16 )
              {
                if ( *(_BYTE *)(v4 + 8) != 80 )
                  return v3;
                unk_4D04170 = 1;
                goto LABEL_34;
              }
              goto LABEL_14;
            }
            if ( (v9[195] & 9) != 1 )
            {
              if ( *(_BYTE *)(v4 + 8) != 80 )
                return v3;
              v48 = v7;
              v18 = sub_736C60(80, a2[13]);
              v11 = v48;
              v12 = 0;
              v54 = v18;
              goto LABEL_57;
            }
LABEL_50:
            sub_684B30(2651, v4 + 56);
            goto LABEL_37;
          }
          v44 = v7;
          v30 = sub_724DC0();
          v31 = (const char *)a2[1];
          v63[0] = v30;
          v60 = strlen(v31) + 1;
          v53 = (char *)sub_724830(v60);
          strcpy(v53, (const char *)a2[1]);
          sub_724C70(v63[0], 2);
          *(_QWORD *)(v63[0] + 128) = sub_73CA60(v60);
          v32 = v63[0];
          *(_QWORD *)(v63[0] + 176) = v60;
          *(_QWORD *)(v32 + 184) = v53;
          v33 = sub_7276D0();
          *(_BYTE *)(v33 + 10) = 3;
          v61 = v33;
          sub_7296C0(&v62);
          v34 = sub_73A460(v63[0]);
          v35 = v62;
          *(_QWORD *)(v61 + 40) = v34;
          sub_729730(v35);
          *(_QWORD *)(v4 + 32) = v61;
          sub_724E30(v63);
          v55 = a2;
          v7 = v44;
LABEL_27:
          v15 = 0;
          if ( unk_4F077A8 <= 0x9FC3u )
          {
LABEL_28:
            v9 = 0;
            goto LABEL_29;
          }
          goto LABEL_44;
        }
        v59 = v7;
        sub_684B30(2745, a1 + 56);
        *(_BYTE *)(a1 + 8) = 0;
        v7 = v59;
      }
      else
      {
        v57 = v7;
        sub_684B30(2744, a1 + 56);
        *(_BYTE *)(a1 + 8) = 0;
        v7 = v57;
      }
      if ( unk_4F077A8 <= 0x9FC3u )
        return v3;
      goto LABEL_43;
  }
  if ( unk_4F077A8 <= 0x9FC3u )
  {
    if ( a3 != 6 )
    {
      if ( *(_BYTE *)(a1 + 8) != 80 )
        return v3;
      v58 = *(_QWORD *)(a1 + 48);
      v29 = sub_736C60(80, a2[13]);
      v11 = v58;
      v54 = v29;
      if ( !v58 )
      {
        v55 = 0;
        v5 = 0;
        v12 = a2;
        v9 = 0;
        goto LABEL_15;
      }
      v55 = 0;
      v12 = a2;
      v5 = 0;
      v9 = 0;
      goto LABEL_88;
    }
    if ( (unsigned __int8)(*((_BYTE *)a2 + 140) - 9) > 2u
      || (*(_BYTE *)(a1 + 11) & 1) != 0
      || (*((_BYTE *)a2 + 177) & 0x10) == 0 )
    {
      v8 = a2;
      v9 = 0;
LABEL_12:
      if ( *(_BYTE *)(a1 + 8) != 80 )
        return v3;
      v40 = v8;
      v5 = 0;
      v45 = *(_QWORD *)(a1 + 48);
      v10 = sub_736C60(80, a2[13]);
      v11 = v45;
      v12 = v40;
      v55 = 0;
      v54 = v10;
LABEL_14:
      if ( !v11 )
        goto LABEL_15;
      if ( a3 != 11 )
      {
LABEL_88:
        if ( a3 == 6 )
          goto LABEL_73;
LABEL_15:
        if ( v4 != v54 )
        {
          v46 = v12;
          sub_684B30(2648, v54 + 56);
          v12 = v46;
          *(_BYTE *)(v54 + 8) = 0;
        }
        goto LABEL_17;
      }
LABEL_57:
      if ( (*(_BYTE *)(v11 + 127) & 0x10) == 0 && (char)v9[202] >= 0 )
        goto LABEL_59;
      goto LABEL_15;
    }
    goto LABEL_37;
  }
  v55 = 0;
  v15 = a2;
  v5 = 0;
  if ( !v7 || a3 != 6 )
    goto LABEL_28;
LABEL_47:
  if ( (unsigned __int8)(*((_BYTE *)v15 + 140) - 9) <= 2u
    && ((*(_BYTE *)(v7 + 130) & 8) != 0 || (*((_BYTE *)v15 + 178) & 1) != 0) )
  {
    goto LABEL_50;
  }
  if ( *(_BYTE *)(v4 + 8) != 80 )
    return v3;
  v43 = v15;
  v9 = 0;
  v51 = v7;
  v26 = sub_736C60(80, a2[13]);
  v11 = v51;
  v12 = v43;
  v54 = v26;
LABEL_73:
  if ( (*(_BYTE *)(v11 + 127) & 0xC0) != 0xC0 )
    goto LABEL_15;
LABEL_59:
  if ( v4 == v54 )
  {
    sub_6854C0(2649, v4 + 56, *a2);
    *(_BYTE *)(v4 + 8) = 0;
    return v3;
  }
  v49 = v12;
  v36 = sub_736C60(80, v54);
  v19 = sub_736C60(80, v4);
  v12 = v49;
  if ( *(_QWORD *)(v19 + 32) )
  {
    v42 = v49;
    v38 = v4;
    v50 = v9;
    v20 = *(_QWORD **)(v19 + 32);
    v39 = v5;
    v21 = *(_QWORD **)(v36 + 32);
    while ( 1 )
    {
      v22 = v20[5];
      if ( !v21 )
        break;
      v23 = v21;
      while ( 1 )
      {
        v24 = v23[5];
        if ( v24 == v22 )
          break;
        v25 = *(_QWORD *)(v22 + 176);
        if ( v25 == *(_QWORD *)(v24 + 176) && !memcmp(*(const void **)(v22 + 184), *(const void **)(v24 + 184), v25) )
          break;
        v23 = (_QWORD *)*v23;
        if ( !v23 )
          goto LABEL_85;
      }
      v20 = (_QWORD *)*v20;
      if ( !v20 )
      {
        v9 = v50;
        v12 = v42;
        v5 = v39;
        v4 = v38;
        v3 = a2;
        goto LABEL_70;
      }
    }
LABEL_85:
    v27 = *(_QWORD *)(v22 + 184);
    v4 = v38;
    v28 = v20 + 3;
    v9 = v50;
    v5 = v39;
    v3 = a2;
    v52 = sub_67D9E0(2647, v28, v27);
    sub_67DDB0(v52, 2650, v36 + 56);
    sub_685910(v52);
    *(_BYTE *)(v38 + 8) = 0;
    v12 = v42;
  }
LABEL_70:
  *(_BYTE *)(v54 + 8) = 0;
LABEL_17:
  if ( *(_BYTE *)(v4 + 8) == 80 )
  {
    unk_4D04170 = 1;
    if ( a3 == 11 )
    {
      v9[201] |= 4u;
      return v3;
    }
    if ( a3 == 7 )
    {
      *((_BYTE *)v5 + 168) |= 0x80u;
      return v3;
    }
    if ( a3 != 28 )
    {
      *((_BYTE *)v12 + 143) |= 0x40u;
      return v3;
    }
LABEL_34:
    *((_BYTE *)v55 + 124) |= 0x20u;
    *(_BYTE *)(unk_4F04C68 + 776LL * unk_4F04C64 + 13) |= 0x80u;
  }
  return v3;
}
