// Function: sub_6944D0
// Address: 0x6944d0
//
__int64 __fastcall sub_6944D0(__int64 a1, __int64 a2)
{
  unsigned int v3; // r15d
  __int64 v4; // rdi
  __int64 v5; // r14
  __int64 v6; // r13
  int v7; // eax
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int16 v11; // bx
  __int64 v12; // r15
  __int64 *v13; // rax
  __int64 v14; // r15
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v18; // rax
  __int64 v19; // r15
  __int64 v20; // r14
  __int64 v21; // rcx
  __int64 v22; // rax
  __int16 v23; // ax
  __int64 v24; // rax
  unsigned int v25; // eax
  __int64 *v26; // [rsp+8h] [rbp-58h]
  __int64 v27; // [rsp+10h] [rbp-50h]
  char v28; // [rsp+1Fh] [rbp-41h]
  _DWORD v29[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v3 = word_4F06418[0];
  v4 = v3;
  v5 = sub_693F30(word_4F06418[0]);
  v6 = sub_6877C0(v3);
  v7 = sub_693EA0(v3);
  if ( v6 )
  {
    if ( (*(_DWORD *)(qword_4D03C50 + 16LL) & 0x400000FF) == 0x40000001 )
    {
      v25 = sub_6E92D0();
      sub_6E6890(v25, a1);
      goto LABEL_13;
    }
    if ( !v7 )
    {
      v11 = v3;
      v12 = qword_4F04C68[0] + 776LL * *(int *)(v6 + 240);
      v13 = *(__int64 **)(v12 + 248);
      if ( !v13 )
      {
        v4 = 24;
        v13 = (__int64 *)sub_823970(24);
        *(_QWORD *)(v12 + 248) = v13;
        *v13 = 0;
        v13[1] = 0;
        v13[2] = 0;
      }
      if ( v11 == 140 )
      {
        v14 = v13[1];
        v26 = v13 + 1;
        if ( v14 )
          goto LABEL_10;
      }
      else if ( v11 > 0x8Cu )
      {
        if ( v11 != 141 )
          goto LABEL_42;
        v14 = v13[2];
        v26 = v13 + 2;
        if ( v14 )
          goto LABEL_10;
      }
      else
      {
        if ( (unsigned __int16)(v11 - 138) > 1u )
          goto LABEL_42;
        v14 = *v13;
        v26 = v13;
        if ( *v13 )
        {
LABEL_10:
          sub_6F8E70(v14, &dword_4F063F8, &qword_4F063F0, a1, 0);
          goto LABEL_13;
        }
      }
      sub_6943F0(0, 17);
      if ( unk_4F04C50 && (v18 = *(_QWORD *)(unk_4F04C50 + 32LL)) != 0 && (*(_BYTE *)(v18 + 198) & 0x10) != 0 )
      {
        v28 = 0;
        if ( dword_4F04C44 == -1 )
          v28 = ((*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) >> 1) ^ 1) & 1;
      }
      else
      {
        v28 = 0;
      }
      sub_7296C0(v29);
      v27 = sub_740630(xmmword_4F06300);
      sub_729730(v29[0]);
      if ( dword_4F04C44 == -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0 )
      {
        v20 = v27;
        v19 = sub_73C570(*(_QWORD *)(v27 + 128), 1, -1);
      }
      else
      {
        v19 = *(_QWORD *)&dword_4D03B80;
        sub_7296C0(v29);
        v20 = sub_724D50(12);
        sub_70FDD0(v27, v20, v19, 0);
        sub_729730(v29[0]);
      }
      v4 = v19;
      if ( v28 )
      {
        v22 = sub_735FB0(v19, 2, 0, v21);
        *(_BYTE *)(v22 + 89) &= ~1u;
        v14 = v22;
        v23 = *(_WORD *)(v22 + 176);
        *(_BYTE *)(v14 + 156) |= 0x21u;
      }
      else
      {
        v24 = sub_735FB0(v19, 2, *(unsigned int *)(v6 + 240), v21);
        *(_BYTE *)(v24 + 89) |= 1u;
        v14 = v24;
        v23 = *(_WORD *)(v24 + 176);
      }
      *(_QWORD *)(v14 + 184) = v20;
      *(_WORD *)(v14 + 176) = v23 & 0xBF | 0x140;
      if ( v11 == 138 )
      {
        if ( dword_4D04964 )
        {
          sub_72A420(v14);
          *v26 = v14;
          goto LABEL_10;
        }
LABEL_31:
        *(_BYTE *)(v14 + 172) |= 8u;
        sub_72A420(v14);
        *v26 = v14;
        goto LABEL_10;
      }
      if ( (unsigned __int16)(v11 - 139) <= 2u )
        goto LABEL_31;
LABEL_42:
      sub_721090(v4);
    }
    sub_6943F0(v7, 17);
LABEL_12:
    sub_6E7020(xmmword_4F06300, a1);
    *(_BYTE *)(a1 + 19) &= ~0x10u;
    goto LABEL_13;
  }
  if ( HIDWORD(qword_4F077B4) )
  {
    sub_6943F0(1, 17);
    goto LABEL_12;
  }
  if ( (unsigned int)sub_6E5430(v3, a2, HIDWORD(qword_4F077B4), v8, v9, v10) )
    sub_6851F0(0x40Cu, v5);
  sub_6E6260(a1);
LABEL_13:
  sub_6E26D0(1, a1);
  return sub_7B8B50(1, a1, v15, v16);
}
