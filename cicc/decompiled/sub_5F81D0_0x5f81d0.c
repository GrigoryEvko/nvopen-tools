// Function: sub_5F81D0
// Address: 0x5f81d0
//
__int64 __fastcall sub_5F81D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r9
  _QWORD *v6; // r10
  __int64 *v7; // r15
  __int64 v8; // r13
  __int64 v9; // r14
  __int64 v10; // rbx
  __int64 v11; // r12
  char v12; // al
  char v13; // di
  int v14; // eax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  char v18; // al
  _QWORD *v19; // rsi
  _QWORD *v20; // rax
  __int64 result; // rax
  char v22; // al
  __int64 *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r14
  _QWORD *v26; // rax
  _QWORD *v27; // rax
  _QWORD *v28; // rax
  _BYTE *v29; // rax
  __int64 v30; // rax
  int v31; // eax
  __int64 v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rax
  bool v35; // zf
  __int64 v36; // rax
  __int64 v37; // rax
  _BOOL4 v38; // [rsp+Ch] [rbp-64h]
  _BOOL4 v39; // [rsp+10h] [rbp-60h]
  char v40; // [rsp+16h] [rbp-5Ah]
  char v41; // [rsp+17h] [rbp-59h]
  __int64 v42; // [rsp+18h] [rbp-58h]
  __int64 v43; // [rsp+18h] [rbp-58h]
  _QWORD *v44; // [rsp+18h] [rbp-58h]
  _QWORD *v45; // [rsp+18h] [rbp-58h]
  _QWORD *v46; // [rsp+18h] [rbp-58h]
  _QWORD *v47; // [rsp+18h] [rbp-58h]
  _QWORD *v48; // [rsp+18h] [rbp-58h]
  _QWORD *v49; // [rsp+18h] [rbp-58h]
  _QWORD *v50; // [rsp+20h] [rbp-50h]
  _QWORD *v51; // [rsp+20h] [rbp-50h]
  __int64 v52; // [rsp+20h] [rbp-50h]
  __int64 v53; // [rsp+20h] [rbp-50h]
  __int64 v54; // [rsp+20h] [rbp-50h]
  __int64 v55; // [rsp+20h] [rbp-50h]
  __int64 v56; // [rsp+20h] [rbp-50h]
  __int64 v57; // [rsp+20h] [rbp-50h]
  __int64 v58; // [rsp+20h] [rbp-50h]
  __int64 v59; // [rsp+28h] [rbp-48h]
  __int64 v60; // [rsp+28h] [rbp-48h]
  __int64 v62; // [rsp+38h] [rbp-38h]

  v5 = a1;
  v6 = (_QWORD *)a2;
  v7 = (__int64 *)(a2 + 48);
  v8 = a3;
  v9 = *(_QWORD *)a2;
  v62 = a4;
  v10 = *(_QWORD *)(a3 + 88);
  v11 = *(_QWORD *)(*(_QWORD *)a2 + 88LL);
  v59 = *(_QWORD *)a1;
  *(_BYTE *)(v11 + 192) |= 2u;
  if ( (*(_BYTE *)(v11 + 193) & 0x10) == 0 )
  {
    v12 = *(_BYTE *)(v10 + 198);
    v42 = a1;
    v50 = (_QWORD *)a2;
    v13 = v12 & 0x18;
    v14 = v12 & 0x10;
    v41 = v14;
    v40 = v13;
    v39 = v14 == 0;
    v38 = v13 == 24;
    v15 = sub_736C60(87, *(_QWORD *)(a2 + 184));
    v6 = (_QWORD *)a2;
    v5 = v42;
    if ( v15 || (a1 = 87, v30 = sub_736C60(87, *(_QWORD *)(a2 + 200)), v6 = (_QWORD *)a2, v5 = v42, v30) )
    {
      a1 = 86;
      v43 = v5;
      v51 = v6;
      v16 = sub_736C60(86, v6[23]);
      v6 = v51;
      v5 = v43;
      if ( v16 )
        goto LABEL_4;
      a1 = 86;
      v34 = sub_736C60(86, v51[25]);
      v6 = v51;
      v5 = v43;
      v35 = v34 == 0;
      v31 = 1;
      a4 = !v35;
    }
    else
    {
      if ( !unk_4D045EC || (a1 = 86, v36 = sub_736C60(86, *(_QWORD *)(a2 + 184)), v6 = (_QWORD *)a2, v5 = v42, v36) )
      {
        v31 = 0;
        a4 = 1;
        goto LABEL_63;
      }
      a2 = *(_QWORD *)(a2 + 200);
      a1 = 86;
      v37 = sub_736C60(86, a2);
      v6 = v50;
      v5 = v42;
      a3 = v37;
      a4 = v37 != 0;
      v31 = unk_4D045EC;
      if ( unk_4D045EC )
      {
        if ( !a3 )
        {
          if ( v41 )
          {
            v31 = 1;
            goto LABEL_90;
          }
LABEL_82:
          a2 = 3542;
          goto LABEL_6;
        }
        v31 = 0;
      }
    }
    a3 = (unsigned int)a4 & v31;
    if ( !(_DWORD)a3 )
    {
      if ( (v31 & v39) != 0 )
        goto LABEL_82;
LABEL_63:
      a3 = v40 == 16;
      if ( ((unsigned int)a4 & (unsigned int)a3) != 0 )
      {
        a2 = 3545;
        goto LABEL_6;
      }
      a2 = 3546;
      if ( ((unsigned int)a4 & v38) != 0 )
        goto LABEL_6;
LABEL_90:
      if ( (v31 & v38) == 0 )
      {
LABEL_7:
        v17 = *(_QWORD *)(v11 + 152);
        if ( *(_BYTE *)(v17 + 140) != 7 )
          goto LABEL_8;
        goto LABEL_56;
      }
      a2 = 3547;
LABEL_6:
      a1 = 8;
      v44 = v6;
      v52 = v5;
      sub_686B60(8, a2, v7, v8, v9);
      v6 = v44;
      v5 = v52;
      goto LABEL_7;
    }
LABEL_4:
    if ( v41 )
    {
      a2 = 3544;
      if ( v40 != 16 )
        goto LABEL_7;
    }
    else
    {
      a2 = 3543;
    }
    goto LABEL_6;
  }
  if ( (*(_BYTE *)(v10 + 193) & 0x10) == 0 )
  {
    a3 = 0x8000000000000LL;
    if ( (*(_QWORD *)(v10 + 200) & 0x8000001000000LL) != 0x8000000000000LL || (*(_BYTE *)(v10 + 192) & 2) != 0 )
    {
      v22 = *(_BYTE *)(v10 + 198);
      a3 = v22 & 0x18;
      if ( (_BYTE)a3 != 16 )
      {
        *(_BYTE *)(v11 + 198) |= 8u;
        v22 = *(_BYTE *)(v10 + 198);
      }
      if ( (v22 & 0x10) != 0 )
        *(_BYTE *)(v11 + 198) |= 0x10u;
    }
  }
  v17 = *(_QWORD *)(v11 + 152);
  if ( *(_BYTE *)(v17 + 140) != 7 )
    goto LABEL_31;
LABEL_56:
  v29 = *(_BYTE **)(*(_QWORD *)(v17 + 168) + 56LL);
  if ( v29 && (*v29 & 2) != 0 )
  {
    a1 = v11;
    v47 = v6;
    v56 = v5;
    sub_5F80E0(v11);
    v6 = v47;
    v5 = v56;
  }
LABEL_8:
  if ( (*(_BYTE *)(v11 + 193) & 0x10) != 0 )
  {
LABEL_31:
    v7 = (__int64 *)(v9 + 48);
LABEL_32:
    if ( *(char *)(v5 + 8) < 0 )
      goto LABEL_11;
    goto LABEL_33;
  }
  if ( (*(_BYTE *)(v10 + 195) & 1) == 0 )
    goto LABEL_32;
  a1 = v8;
  v45 = v6;
  v53 = v5;
  sub_894C00(v8, a2, a3, a4, a5);
  v5 = v53;
  v6 = v45;
  if ( *(char *)(v53 + 8) < 0 )
    goto LABEL_11;
LABEL_33:
  if ( (*(_BYTE *)(v11 + 195) & 8) == 0 )
  {
    if ( dword_4F077C4 == 2 && unk_4F07778 > 201702 && (*(_BYTE *)(*(_QWORD *)(v9 + 88) + 206LL) & 0x10) != 0 )
    {
      v18 = *(_BYTE *)(*(_QWORD *)(v8 + 88) + 206LL) & 0x10;
      goto LABEL_54;
    }
    v23 = (__int64 *)qword_4CF7FD0;
    if ( qword_4CF7FD0 )
    {
      qword_4CF7FD0 = *(_QWORD *)qword_4CF7FD0;
    }
    else
    {
      a1 = 40;
      v49 = v6;
      v58 = v5;
      v23 = (__int64 *)sub_823970(40);
      v6 = v49;
      v5 = v58;
    }
    v24 = qword_4CF7FC8;
    v23[1] = v9;
    v23[2] = v8;
    *v23 = v24;
    v23[3] = 0;
    qword_4CF7FC8 = (__int64)v23;
    v23[4] = *v7;
  }
LABEL_11:
  a3 = *(_BYTE *)(*(_QWORD *)(v9 + 88) + 206LL) & 0x10;
  v18 = *(_BYTE *)(*(_QWORD *)(v8 + 88) + 206LL) & 0x10;
  if ( !(_BYTE)a3 )
  {
    if ( v18 )
    {
      a1 = 1789;
      v48 = v6;
      v57 = v5;
      sub_6854C0(1789, v7, v8);
      v6 = v48;
      v5 = v57;
    }
    goto LABEL_14;
  }
LABEL_54:
  if ( !v18 )
  {
    a1 = 1788;
    v46 = v6;
    v55 = v5;
    sub_6854C0(1788, v7, v8);
    v5 = v55;
    v6 = v46;
  }
LABEL_14:
  if ( (*(_BYTE *)(v10 + 192) & 0x10) != 0 )
  {
LABEL_20:
    sub_6854C0(1850, v7, v8);
    goto LABEL_21;
  }
  v19 = *(_QWORD **)(v62 + 112);
  if ( v19 )
  {
    while ( 1 )
    {
      a3 = v19[2];
      v20 = (_QWORD *)v19[1];
      a4 = *(_QWORD *)a3;
      if ( v20 != *(_QWORD **)a3 )
        break;
LABEL_40:
      v19 = (_QWORD *)*v19;
      if ( !v19 )
        goto LABEL_41;
    }
    while ( 1 )
    {
      a3 = *(_QWORD *)(v20[2] + 40LL);
      if ( (*(_BYTE *)(a3 + 176) & 1) != 0 )
        goto LABEL_20;
      v20 = (_QWORD *)*v20;
      if ( v20 == (_QWORD *)a4 )
        goto LABEL_40;
    }
  }
LABEL_41:
  v25 = *(_QWORD *)(*v6 + 88LL);
  v26 = *(_QWORD **)(v62 + 120);
  do
  {
    if ( !v26 )
    {
      v54 = v5;
      v27 = (_QWORD *)sub_725050(a1, v19, a3, a4);
      v27[2] = v10;
      v27[1] = v25;
      v27[4] = a5;
      sub_5E4860(v62, v27);
      v5 = v54;
      goto LABEL_45;
    }
    a3 = v26[2];
    a4 = (__int64)v26;
    v26 = (_QWORD *)*v26;
  }
  while ( v10 != a3 );
  *(_QWORD *)(a4 + 8) = v25;
  *(_QWORD *)(a4 + 24) = 0;
  for ( *(_QWORD *)(a4 + 32) = a5; v26; *(_QWORD *)a4 = v26 )
  {
    if ( v10 != v26[2] )
      break;
    v26 = (_QWORD *)*v26;
  }
LABEL_45:
  if ( a5 )
  {
    *(_BYTE *)(v11 + 192) |= 0x40u;
    v28 = (_QWORD *)qword_4CF7FC0;
    if ( qword_4CF7FC0 )
    {
      qword_4CF7FC0 = *(_QWORD *)qword_4CF7FC0;
    }
    else
    {
      v60 = v5;
      v28 = (_QWORD *)sub_823970(40);
      v5 = v60;
    }
    *v28 = 0;
    v28[3] = v10;
    v28[1] = v62;
    v28[2] = a5;
    v28[4] = v11;
    if ( *(_QWORD *)(v5 + 40) )
      **(_QWORD **)(v5 + 48) = v28;
    else
      *(_QWORD *)(v5 + 40) = v28;
    *(_QWORD *)(v5 + 48) = v28;
  }
  else
  {
    v32 = *(_QWORD *)(*(_QWORD *)(v59 + 168) + 24LL);
    if ( v32 )
    {
      if ( v62 == v32 )
      {
LABEL_77:
        *(_WORD *)(v11 + 224) = *(_WORD *)(v10 + 224);
      }
      else
      {
        while ( 1 )
        {
          v33 = sub_5EBA50(v32);
          v32 = v33;
          if ( !v33 )
            break;
          if ( v62 == v33 )
            goto LABEL_77;
        }
      }
    }
  }
LABEL_21:
  result = *(_BYTE *)(v10 + 193) & 4;
  if ( (*(_BYTE *)(v11 + 193) & 4) != 0 )
  {
    if ( !(_BYTE)result )
      return sub_6854C0(2935, v7, v8);
  }
  else if ( (_BYTE)result )
  {
    return sub_6854C0(2936, v7, v8);
  }
  return result;
}
