// Function: sub_6968F0
// Address: 0x6968f0
//
__int64 __fastcall sub_6968F0(__int64 a1, _DWORD *a2)
{
  char v3; // al
  char v4; // al
  __int64 v5; // rdi
  __int64 v7; // r13
  char v8; // al
  __int64 v9; // rax
  char i; // dl
  char v11; // dl
  __int64 v12; // rdi
  unsigned __int8 v13; // al
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19[5]; // [rsp+8h] [rbp-28h] BYREF

  *a2 = 0;
  v3 = *(_BYTE *)(a1 + 16);
  v19[0] = 0;
  if ( v3 == 1 )
  {
    v7 = sub_6E36E0(*(_QWORD *)(a1 + 144), 0);
  }
  else
  {
    if ( v3 != 2 || (v15 = sub_72ECB0(a1 + 144)) == 0 )
    {
LABEL_3:
      if ( (*(_BYTE *)(a1 + 18) & 0x28) != 8 )
        goto LABEL_4;
      goto LABEL_22;
    }
    v7 = sub_6E36E0(v15, 0);
  }
  if ( !v7 )
    goto LABEL_3;
  v8 = *(_BYTE *)(v7 + 24);
  if ( v8 != 1
    || (*(_BYTE *)(a1 + 20) & 2) != 0
    || (v11 = *(_BYTE *)(v7 + 56), (unsigned __int8)(v11 - 94) > 1u) && (unsigned __int8)(v11 - 100) > 1u )
  {
    if ( (*(_BYTE *)(a1 + 18) & 0x28) != 8 )
      goto LABEL_4;
    goto LABEL_28;
  }
  v12 = *(_QWORD *)(*(_QWORD *)(v7 + 72) + 16LL);
  *a2 = 1;
  v13 = *(_BYTE *)(v7 + 56);
  if ( v13 <= 0x5Fu )
  {
    if ( v13 <= 0x5Du )
      goto LABEL_42;
    v5 = *(_QWORD *)(*(_QWORD *)(v12 + 56) + 120LL);
    v19[0] = v5;
LABEL_17:
    if ( !(unsigned int)sub_8D2B80(v5) )
      return v19[0];
    goto LABEL_14;
  }
  if ( (unsigned __int8)(v13 - 100) > 1u )
LABEL_42:
    sub_721090(v12);
  v17 = sub_6E36E0(v12, 0);
  v7 = v17;
  if ( v17 )
  {
    v8 = *(_BYTE *)(v17 + 24);
LABEL_28:
    switch ( v8 )
    {
      case 3:
        v14 = *(_QWORD *)(v7 + 56);
        if ( (*(_BYTE *)(v14 + 170) & 1) != 0 )
        {
          v18 = sub_657CE0(v14);
          v14 = *(_QWORD *)(v7 + 56);
          v19[0] = v18;
        }
        else if ( (*(_BYTE *)(v14 + 176) & 0x20) != 0 )
        {
          v16 = sub_73D4C0(*(_QWORD *)(v14 + 120), dword_4F077C4 == 2);
          v14 = *(_QWORD *)(v7 + 56);
          v19[0] = v16;
        }
        else
        {
          v19[0] = *(_QWORD *)(v14 + 120);
        }
        if ( (*(_WORD *)(v14 + 156) & 0x101) == 0x101 )
          sub_6851C0(0xDF8u, (_DWORD *)(a1 + 68));
        v5 = v19[0];
        goto LABEL_32;
      case 20:
        v5 = *(_QWORD *)(*(_QWORD *)(v7 + 56) + 152LL);
        v19[0] = v5;
        goto LABEL_32;
      case 24:
        v5 = *(_QWORD *)v7;
        v19[0] = *(_QWORD *)v7;
        goto LABEL_32;
    }
    goto LABEL_4;
  }
LABEL_22:
  if ( *(_WORD *)(a1 + 16) == 514 )
  {
    v5 = *(_QWORD *)(a1 + 272);
    v19[0] = v5;
LABEL_32:
    *a2 = 1;
    goto LABEL_17;
  }
LABEL_4:
  if ( !dword_4F077BC
    || (_DWORD)qword_4F077B4
    || qword_4F077A8 > 0x1FBCFu
    || (*(_DWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4) & 0x8200FF) != 0x20009 )
  {
    v19[0] = *(_QWORD *)a1;
    if ( (unsigned int)sub_6E9E70(a1) )
    {
      if ( *(_BYTE *)(a1 + 16) )
      {
        v9 = *(_QWORD *)a1;
        for ( i = *(_BYTE *)(*(_QWORD *)a1 + 140LL); i == 12; i = *(_BYTE *)(v9 + 140) )
          v9 = *(_QWORD *)(v9 + 160);
        if ( i )
          goto LABEL_13;
      }
    }
    else
    {
      v4 = *(_BYTE *)(a1 + 17);
      if ( v4 == 1 )
      {
        if ( !(unsigned int)sub_6ED0A0(a1) )
          goto LABEL_12;
        v4 = *(_BYTE *)(a1 + 17);
      }
      if ( v4 == 3 && *(_BYTE *)(a1 + 16) != 3 )
      {
LABEL_12:
        if ( !(unsigned int)sub_696840(a1) )
        {
          v19[0] = sub_72D600(v19[0]);
          v5 = v19[0];
          goto LABEL_17;
        }
LABEL_13:
        v19[0] = *(_QWORD *)&dword_4D03B80;
        if ( !(unsigned int)sub_8D2B80(*(_QWORD *)&dword_4D03B80) )
          return v19[0];
LABEL_14:
        sub_73C7D0(v19);
        return v19[0];
      }
      if ( (unsigned int)sub_6ED0A0(a1) )
      {
        if ( !(unsigned int)sub_696840(a1) )
        {
          v19[0] = sub_72D6A0(v19[0]);
          v5 = v19[0];
          goto LABEL_17;
        }
        goto LABEL_13;
      }
    }
    v5 = v19[0];
    goto LABEL_17;
  }
  return *(_QWORD *)&dword_4D03B80;
}
