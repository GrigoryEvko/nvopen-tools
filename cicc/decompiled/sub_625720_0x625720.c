// Function: sub_625720
// Address: 0x625720
//
__int64 __fastcall sub_625720(__int64 a1)
{
  bool v2; // bl
  __int64 result; // rax
  __int64 v4; // rcx
  char v5; // dl
  char v6; // si
  __int64 v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // rax
  char i; // dl
  char v11; // dl
  char v12; // cl
  int v13; // edi
  unsigned int v14; // [rsp+Ch] [rbp-24h]

  v2 = (*(_BYTE *)(a1 + 125) & 4) != 0;
  result = sub_8D3410(*(_QWORD *)(a1 + 280));
  if ( (_DWORD)result )
  {
    if ( (*(_BYTE *)(a1 + 123) & 2) != 0 )
    {
      result = *(_QWORD *)(a1 + 424);
      if ( result )
      {
        if ( (*(_BYTE *)(result + 131) & 8) != 0 )
          goto LABEL_9;
        result = (__int64)&dword_4D04490;
        if ( dword_4D04490 )
          goto LABEL_9;
      }
    }
    if ( !v2 )
    {
      sub_6851C0(1589, a1 + 104);
      return sub_643D80(a1);
    }
    goto LABEL_4;
  }
  v4 = *(_QWORD *)(a1 + 280);
  v5 = *(_BYTE *)(v4 + 140);
  if ( (*(_BYTE *)(a1 + 125) & 2) == 0 && !v2 )
  {
    if ( v5 != 7 || (*(_BYTE *)(a1 + 123) & 0x10) != 0 )
      goto LABEL_9;
    goto LABEL_26;
  }
  if ( v5 == 7 )
  {
    if ( (*(_BYTE *)(a1 + 123) & 0x10) != 0 )
    {
      v6 = *(_BYTE *)(a1 + 122);
      if ( (v6 & 0x50) == 0 || (v12 = *(_BYTE *)(*(_QWORD *)(v4 + 160) + 140LL), v12 != 6) && v12 != 13 )
      {
LABEL_17:
        if ( !v2 )
          goto LABEL_9;
        goto LABEL_18;
      }
      if ( v2 )
      {
        v6 = *(_BYTE *)(a1 + 122);
        result = v2;
        goto LABEL_18;
      }
    }
    else
    {
      v11 = *(_BYTE *)(*(_QWORD *)(v4 + 160) + 140LL);
      if ( v11 != 6 && v11 != 13 )
        goto LABEL_26;
      if ( v2 )
      {
LABEL_37:
        v9 = *(_QWORD *)(a1 + 272);
        for ( i = *(_BYTE *)(v9 + 140); i == 12; i = *(_BYTE *)(v9 + 140) )
          v9 = *(_QWORD *)(v9 + 160);
        if ( i )
        {
          if ( !v2 )
          {
            v13 = -(unk_4D04430 == 0);
            LOBYTE(v13) = v13 & 0x14;
            sub_6851C0((unsigned int)(v13 + 1826), a1 + 104);
            return sub_643D80(a1);
          }
        }
        else if ( !v2 )
        {
          return sub_643D80(a1);
        }
        v6 = *(_BYTE *)(a1 + 122);
        result = 1;
        goto LABEL_18;
      }
    }
    sub_6851C0(2652, a1 + 104);
    if ( (*(_BYTE *)(a1 + 123) & 0x10) != 0 )
      return sub_643D80(a1);
    result = 1;
LABEL_26:
    if ( unk_4F0774C && (*(_WORD *)(a1 + 122) & 0x220) == 0 && !v2 )
    {
      *(_BYTE *)(a1 + 125) |= 8u;
      if ( unk_4F07748 )
      {
        v14 = result;
        sub_684B30(2730, a1 + 104);
        result = v14;
      }
      if ( !(_DWORD)result )
        goto LABEL_9;
      return sub_643D80(a1);
    }
    goto LABEL_37;
  }
  v6 = *(_BYTE *)(a1 + 122);
  if ( (v6 & 0x50) == 0 || v5 != 6 && v5 != 13 )
    goto LABEL_17;
  result = v2;
  if ( !v2 )
  {
    sub_6851C0(2652, a1 + 104);
    return sub_643D80(a1);
  }
LABEL_18:
  if ( (v6 & 0x40) != 0
    || (*(_BYTE *)(a1 + 132) & 4) != 0 && (dword_4F077C4 != 2 || unk_4F07778 <= 202001)
    || (_DWORD)result )
  {
LABEL_4:
    sub_625700(a1);
    return sub_643D80(a1);
  }
LABEL_9:
  if ( *(char *)(a1 + 121) < 0 )
  {
    result = (*(_BYTE *)(a1 + 123) & 0x10) != 0;
    if ( (*(_QWORD *)(a1 + 312) == 0) != (_BYTE)result )
    {
      v7 = a1 + 104;
      v8 = 5;
      if ( dword_4D04964 )
        v8 = unk_4F07471;
      return sub_684AA0(v8, 2409, v7);
    }
  }
  return result;
}
