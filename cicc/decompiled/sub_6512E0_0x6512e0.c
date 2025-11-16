// Function: sub_6512E0
// Address: 0x6512e0
//
__int64 __fastcall sub_6512E0(__int64 a1, int a2, int a3, int a4, int a5, int a6)
{
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r8
  char v11; // al
  __int64 v13; // rdx
  _BYTE *v14; // rax
  _BOOL4 v15; // eax
  int v16; // [rsp+4h] [rbp-4Ch]
  __int64 v17; // [rsp+8h] [rbp-48h]
  __int64 v18; // [rsp+8h] [rbp-48h]
  char v19[52]; // [rsp+1Ch] [rbp-34h] BYREF

  if ( (_DWORD)a1 )
    a1 = 256;
  if ( unk_4D04808 | a2 )
    a1 = (unsigned int)a1 | 1;
  if ( a4 )
    a1 = (unsigned int)a1 | 0x4000000;
  if ( dword_4F077C4 == 2 )
  {
    if ( word_4F06418[0] != 1 || (unk_4D04A11 & 2) == 0 )
    {
      v16 = a5;
      if ( !(unsigned int)sub_7C0F00(a1, 0) )
        return 0;
      a1 = (unsigned int)a1;
      a5 = v16;
    }
  }
  else if ( word_4F06418[0] != 1 )
  {
    return 0;
  }
  if ( (unk_4D04A10 & 0x18) != 0 )
    return 0;
  v8 = qword_4D04A00;
  v9 = sub_7BF130(a1, 2 * (unsigned int)(a5 == 0), v19);
  v10 = v9;
  if ( !v9 )
    return 0;
  v11 = *(_BYTE *)(v9 + 80);
  if ( a2 )
  {
    if ( v11 == 3 )
      return v10;
LABEL_13:
    if ( dword_4F077C4 != 2 )
    {
LABEL_14:
      if ( !a6 )
        goto LABEL_16;
      if ( v11 != 22 )
        goto LABEL_16;
      v18 = v10;
      v15 = sub_651150(1);
      v10 = v18;
      if ( !v15 )
        goto LABEL_16;
      goto LABEL_36;
    }
    goto LABEL_46;
  }
  if ( !unk_4D04808 || (unk_4D04A12 & 1) != 0 )
    goto LABEL_35;
  if ( unk_4D04860 )
  {
    if ( v11 == 19 )
    {
      v13 = *(_QWORD *)(v10 + 88);
      goto LABEL_30;
    }
LABEL_35:
    if ( v11 == 3 )
      goto LABEL_36;
    goto LABEL_13;
  }
  if ( v11 != 19 )
    goto LABEL_35;
  v13 = *(_QWORD *)(v10 + 88);
  if ( (*(_BYTE *)(v13 + 265) & 1) != 0 )
  {
    if ( dword_4F077C4 != 2 )
    {
LABEL_16:
      if ( (unk_4D04A11 & 0x40) == 0 )
      {
        unk_4D04A10 &= ~0x80u;
        unk_4D04A18 = 0;
      }
      qword_4D04A00 = v8;
      return 0;
    }
LABEL_46:
    if ( (unsigned __int8)(v11 - 4) > 2u )
      goto LABEL_14;
LABEL_36:
    if ( a3 | a2 )
      return v10;
    goto LABEL_37;
  }
LABEL_30:
  if ( (*(_BYTE *)(v13 + 266) & 1) != 0 )
    v10 = *(_QWORD *)(v13 + 200);
  v10 = *(_QWORD *)sub_72EF10(v10, &dword_4F063F8);
  if ( !a3 )
  {
    if ( v10 )
    {
LABEL_37:
      if ( dword_4F04C64 != -1 )
      {
        v14 = (_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
        if ( (v14[7] & 1) != 0 && (dword_4F04C44 != -1 || (v14[6] & 6) != 0 || v14[4] == 12) )
        {
          v17 = v10;
          sub_867130(v10, &dword_4F063F8, 0, 0);
          return v17;
        }
      }
      return v10;
    }
    return 0;
  }
  return v10;
}
