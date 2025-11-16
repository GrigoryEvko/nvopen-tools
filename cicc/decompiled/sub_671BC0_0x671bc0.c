// Function: sub_671BC0
// Address: 0x671bc0
//
_QWORD *__fastcall sub_671BC0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, _QWORD *a5, __int64 a6)
{
  _QWORD *v6; // r13
  int v7; // r12d
  unsigned int v9; // r14d
  __int64 v10; // rdi
  __int64 v11; // rdx
  unsigned __int64 v13; // rsi
  int v14; // eax
  __int64 v15; // rcx
  unsigned __int64 v16; // r9
  char v17; // al
  unsigned __int64 v18; // r10
  __int64 v19; // rax
  unsigned __int64 v20; // rdi
  unsigned __int64 *v21; // rax
  unsigned __int64 v22; // r11
  unsigned __int64 v23; // [rsp+0h] [rbp-240h]
  unsigned __int64 v24; // [rsp+0h] [rbp-240h]
  _QWORD *v25; // [rsp+8h] [rbp-238h]
  unsigned __int64 v26; // [rsp+8h] [rbp-238h]
  _QWORD *v27; // [rsp+10h] [rbp-230h]
  char v28; // [rsp+18h] [rbp-228h]
  unsigned __int64 v29; // [rsp+18h] [rbp-228h]
  unsigned __int64 v30; // [rsp+18h] [rbp-228h]
  __int64 v31; // [rsp+18h] [rbp-228h]
  unsigned int v32; // [rsp+2Ch] [rbp-214h] BYREF
  _QWORD v33[59]; // [rsp+30h] [rbp-210h] BYREF
  __int64 v34; // [rsp+208h] [rbp-38h] BYREF

  v6 = (_QWORD *)a2;
  v7 = (int)a3;
  *(_QWORD *)a2 = 0;
  v27 = (_QWORD *)a1;
  v25 = a5;
  LOBYTE(a3) = (_DWORD)a3 == 0 && unk_4D04808 != 0;
  v28 = (char)a3;
  if ( (_BYTE)a3 )
  {
    if ( !a5 )
    {
      a3 = v33;
      v25 = v33;
      memset(v33, 0, sizeof(v33));
      a1 = (__int64)&v34;
      a4 = 0;
      v33[19] = v33;
      v33[3] = *(_QWORD *)&dword_4F063F8;
      if ( dword_4F077BC )
      {
        if ( qword_4F077A8 <= 0x9F5Fu )
          BYTE2(v33[22]) |= 1u;
      }
    }
  }
  if ( unk_4F04C48 == -1 || (a3 = qword_4F04C68, (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) == 0) )
  {
    if ( dword_4F04C44 == -1 && (dword_4F077C4 != 2 || unk_4F07778 <= 201102 && !dword_4F07774) )
    {
      a1 = 4;
      if ( dword_4D04964 )
        a1 = unk_4F07471;
      a2 = 761;
      sub_684AC0(a1, 761);
    }
  }
  v9 = (v28 & 1) + 2048;
  if ( a6 )
    *(_QWORD *)(a6 + 40) = qword_4F063F0;
  sub_7B8B50(a1, a2, a3, a4);
  if ( dword_4F077C4 == 2 )
  {
    if ( word_4F06418[0] == 1 && (unk_4D04A11 & 2) != 0 )
      goto LABEL_14;
    sub_7C0F00(v9, 0);
  }
  if ( word_4F06418[0] != 1 )
  {
    v10 = 40;
    sub_6851D0(40);
LABEL_10:
    v11 = sub_72C930(v10);
    goto LABEL_11;
  }
LABEL_14:
  v10 = v9;
  v32 = 0;
  v13 = v7 == 0 ? 6 : 10;
  v14 = sub_7C8410(v9, v13, &v32);
  v11 = 0;
  if ( v14 && (unk_4D04A10 & 1) != 0 )
  {
    v13 = v32;
    if ( v32 )
      goto LABEL_17;
    v16 = unk_4D04A18;
    if ( dword_4F077C4 == 2 && unk_4D04A18 && (*(_DWORD *)(unk_4D04A18 + 80LL) & 0x41000) != 0 )
    {
      v23 = unk_4D04A18;
      sub_8841F0(&qword_4D04A00, 0, 0, 0);
      v16 = v23;
    }
    v17 = *(_BYTE *)(v16 + 80);
    v18 = v16;
    if ( v17 == 16 )
    {
      v18 = **(_QWORD **)(v16 + 88);
      v17 = *(_BYTE *)(v18 + 80);
    }
    if ( v17 == 24 )
    {
      v18 = *(_QWORD *)(v18 + 88);
      v17 = *(_BYTE *)(v18 + 80);
    }
    if ( v28 && (unk_4D04A12 & 1) == 0 )
    {
      if ( !unk_4D04860 )
      {
        v22 = v18;
        if ( v17 != 19 )
          goto LABEL_57;
        v19 = *(_QWORD *)(v18 + 88);
        if ( (*(_BYTE *)(v19 + 265) & 1) != 0 )
        {
LABEL_63:
          v13 = v16;
          v10 = 757;
          sub_6854E0(757, v16);
          v11 = 0;
          goto LABEL_17;
        }
        goto LABEL_39;
      }
      if ( v17 == 19 )
      {
        v19 = *(_QWORD *)(v18 + 88);
LABEL_39:
        v20 = v18;
        if ( (*(_BYTE *)(v19 + 266) & 1) != 0 )
          v20 = *(_QWORD *)(v19 + 200);
        v24 = v18;
        v29 = v16;
        v21 = (unsigned __int64 *)sub_72EF10(v20, &dword_4F063F8);
        v16 = v29;
        v18 = v24;
        v22 = *v21;
        if ( *(_DWORD *)(v21[21] + 28) == -2 )
        {
          v25[38] = v21;
          *((_WORD *)v25 + 62) |= 0x480u;
          v25[13] = *(_QWORD *)&dword_4F063F8;
        }
        v17 = *(_BYTE *)(v22 + 80);
LABEL_57:
        if ( v17 == 3 || dword_4F077C4 == 2 && (unsigned __int8)(v17 - 4) <= 2u )
        {
          v13 = v18;
          v10 = 4;
          v26 = v16;
          v30 = v22;
          sub_8767A0(4, v18, &qword_4D04A08, 1);
          v11 = *(_QWORD *)(v30 + 88);
          *v6 = v26;
          goto LABEL_17;
        }
        goto LABEL_63;
      }
    }
    v22 = v18;
    goto LABEL_57;
  }
  v15 = v32;
  if ( !v32 )
  {
    v13 = (unsigned __int64)dword_4F07508;
    v10 = 937;
    sub_6851C0(937, dword_4F07508);
    v11 = 0;
  }
LABEL_17:
  if ( a6 )
    *(_QWORD *)(a6 + 40) = qword_4F063F0;
  if ( !v7 )
  {
    v31 = v11;
    sub_7B8B50(v10, v13, v11, v15);
    v11 = v31;
  }
  if ( !v11 )
    goto LABEL_10;
LABEL_11:
  *v27 = v11;
  return v27;
}
