// Function: sub_86F030
// Address: 0x86f030
//
void sub_86F030()
{
  int v0; // ecx
  __int64 v1; // rsi
  int v2; // edx
  __int64 v3; // rbx
  int v4; // eax
  int v5; // r13d
  int v6; // r9d
  int v7; // r8d
  int v8; // edi
  int v9; // r9d
  __int64 v10; // r15
  int v11; // r8d
  int v12; // edi
  unsigned int v13; // r14d
  char v14; // al
  __int64 v15; // rdi
  _DWORD *v16; // rdx
  unsigned __int8 v17; // di
  unsigned __int8 v18; // cl
  unsigned __int8 v19; // dl
  bool v20; // al
  int v21; // eax
  __int64 v22; // r13
  __int64 v23; // rsi
  int v24; // ebx
  __int64 v25; // r12
  int v26; // ecx
  int v27; // eax
  __int64 v28; // rax
  __int64 v29; // rdi
  char v30; // di
  char v31; // si
  int v32; // r8d
  int v33; // edi
  _DWORD *v34; // rdx
  unsigned __int8 v35; // di
  int v36; // r8d
  int v37; // edi
  int v38; // eax

  v0 = HIDWORD(qword_4F5FD78);
  v1 = unk_4D03B90;
  v2 = qword_4F5FD78;
  v3 = qword_4D03B98 + 176LL * unk_4D03B90;
  v4 = dword_4F5FD80;
  v5 = *(_DWORD *)v3;
  v6 = *(_DWORD *)(v3 + 116);
  *(_QWORD *)(v3 + 48) = 0;
  v7 = *(_DWORD *)(v3 + 120);
  v8 = *(_DWORD *)(v3 + 124);
  *(_QWORD *)(v3 + 56) = 0;
  v9 = v2 | v6;
  v10 = *(_QWORD *)(v3 + 8);
  v11 = v0 | v7;
  v12 = v4 | v8;
  v13 = v5 & 0xFFFFFFFD;
  *(_DWORD *)(v3 + 116) = v9;
  *(_DWORD *)(v3 + 120) = v11;
  *(_DWORD *)(v3 + 124) = v12;
  if ( (v5 & 0xFFFFFFFD) == 4 )
  {
    v1 = *(unsigned __int8 *)(v10 + 40);
    if ( (unsigned __int8)(v1 - 12) > 1u && (_BYTE)v1 != 5 )
    {
LABEL_33:
      v26 = *(_DWORD *)(v3 + 108) | v0;
      v27 = *(_DWORD *)(v3 + 112) | v4;
      LODWORD(qword_4F5FD78) = *(_DWORD *)(v3 + 104) | v2;
      HIDWORD(qword_4F5FD78) = v26;
      dword_4F5FD80 = v27;
      goto LABEL_8;
    }
    goto LABEL_5;
  }
  if ( v13 != 5 )
  {
    if ( v5 == 3 )
    {
      if ( !*(_QWORD *)(*(_QWORD *)(v10 + 80) + 8LL) )
      {
        v36 = *(_DWORD *)(v3 + 108) | v11;
        v37 = *(_DWORD *)(v3 + 112) | v12;
        *(_DWORD *)(v3 + 116) = *(_DWORD *)(v3 + 104) | v9;
        *(_DWORD *)(v3 + 120) = v36;
        *(_DWORD *)(v3 + 124) = v37;
      }
      v34 = *(_DWORD **)(v3 + 168);
      if ( v34 )
      {
        v35 = 8;
        if ( !(_DWORD)qword_4F077B4 )
          v35 = unk_4F07471;
        sub_684AA0(v35, 0xAFEu, v34);
        LODWORD(v1) = unk_4D03B90;
      }
      goto LABEL_29;
    }
    if ( v5 != 1 )
    {
      if ( v5 != 2 || *(_QWORD *)(*(_QWORD *)(v10 + 72) + 8LL) )
        goto LABEL_29;
      if ( !sub_86BC00(*(_QWORD *)(v10 + 48), v1) )
      {
        v38 = *(_DWORD *)(v3 + 112) | *(_DWORD *)(v3 + 124);
        *(_QWORD *)(v3 + 116) |= *(_QWORD *)(v3 + 104);
        *(_DWORD *)(v3 + 124) = v38;
      }
      goto LABEL_56;
    }
    if ( !*(_QWORD *)(v10 + 80) )
    {
      if ( *(_QWORD *)(v10 + 48) )
      {
        if ( sub_86BC00(*(_QWORD *)(v10 + 48), v1) )
        {
LABEL_56:
          LODWORD(v1) = unk_4D03B90;
          goto LABEL_29;
        }
        v9 = *(_DWORD *)(v3 + 116);
        v11 = *(_DWORD *)(v3 + 120);
        v12 = *(_DWORD *)(v3 + 124);
        LODWORD(v1) = unk_4D03B90;
      }
      v32 = *(_DWORD *)(v3 + 108) | v11;
      v33 = *(_DWORD *)(v3 + 112) | v12;
      *(_DWORD *)(v3 + 116) = *(_DWORD *)(v3 + 104) | v9;
      *(_DWORD *)(v3 + 120) = v32;
      *(_DWORD *)(v3 + 124) = v33;
    }
LABEL_29:
    qword_4F5FD78 = *(_QWORD *)(v3 + 116);
    dword_4F5FD80 = *(_DWORD *)(v3 + 124);
    goto LABEL_13;
  }
  v14 = *(_BYTE *)(v10 + 40);
  if ( (unsigned __int8)(v14 - 12) > 1u && v14 != 5 )
    goto LABEL_30;
LABEL_5:
  v15 = *(_QWORD *)(v10 + 48);
  if ( v15 && !sub_86BC00(v15, v1) )
  {
LABEL_30:
    if ( v13 != 4 && v5 != 7 )
      goto LABEL_8;
    v2 = qword_4F5FD78;
    v0 = HIDWORD(qword_4F5FD78);
    v4 = dword_4F5FD80;
    goto LABEL_33;
  }
  qword_4F5FD78 = 0;
  dword_4F5FD80 = 0;
LABEL_8:
  v16 = *(_DWORD **)(v3 + 168);
  if ( v16 )
  {
    v17 = 8;
    if ( !(_DWORD)qword_4F077B4 )
      v17 = unk_4F07471;
    sub_684AA0(v17, 0xAFEu, v16);
  }
  LODWORD(v1) = unk_4D03B90;
LABEL_13:
  if ( (int)v1 > 0 )
  {
    v18 = *(_BYTE *)(v3 + 5);
    v19 = *(_BYTE *)(v3 - 171);
    v20 = ((v19 | v18) & 0x20) != 0;
    if ( v5 )
    {
      v30 = *(_BYTE *)(v3 + 4) & 0x80;
      v31 = *(_BYTE *)(v3 - 172) & 0x7F;
      *(_BYTE *)(v3 - 171) = v19 & 0xDF | (32 * v20);
      *(_BYTE *)(v3 - 172) = v30 | v31;
      if ( v5 == 3 )
        goto LABEL_20;
    }
    else
    {
      if ( *(_QWORD *)(*(_QWORD *)(v10 + 80) + 16LL) )
        *(_BYTE *)(v3 - 172) = *(_BYTE *)(v3 + 4) & 0x80 | *(_BYTE *)(v3 - 172) & 0x7F;
      *(_BYTE *)(v3 - 171) = v19 & 0xDF | (32 * v20);
    }
    *(_BYTE *)(v3 - 171) |= v18 & 0x40;
  }
  if ( v5 )
  {
LABEL_20:
    v21 = unk_4D03B90;
    goto LABEL_21;
  }
  v28 = sub_86B2C0(5);
  sub_86CBE0(v28);
  v29 = qword_4F06BC0;
  if ( *(_QWORD *)(v3 + 128) != qword_4F06BC0 )
    *(_QWORD *)(v3 + 128) = qword_4F06BC0;
  sub_86BFC0(v29);
  v21 = unk_4D03B90;
  if ( unk_4D03B90 > 0 )
  {
    *(_QWORD *)(v3 - 8) = *(_QWORD *)(v3 + 168);
    goto LABEL_20;
  }
LABEL_21:
  v22 = *(_QWORD *)(v3 + 64);
  v23 = *(_QWORD *)(v3 + 72);
  unk_4D03B90 = v21 - 1;
  if ( v22 )
  {
    v24 = dword_4F5FD80;
    v25 = qword_4F5FD78;
    sub_86EEF0(v22, v23);
    if ( (*(_BYTE *)(v22 + 120) & 4) != 0 )
    {
      qword_4F5FD78 = v25;
      dword_4F5FD80 = v24;
    }
  }
}
