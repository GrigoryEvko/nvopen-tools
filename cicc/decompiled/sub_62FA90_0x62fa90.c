// Function: sub_62FA90
// Address: 0x62fa90
//
__int64 __fastcall sub_62FA90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  char i; // dl
  __int64 *v6; // rdx
  __int64 result; // rax
  __int64 v8; // rdx
  char v9; // cl
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  _BYTE v16[40]; // [rsp+8h] [rbp-28h] BYREF

  if ( HIDWORD(qword_4F077B4) && *(_BYTE *)(a1 + 8) == 1 )
  {
    if ( !*(_QWORD *)(a1 + 24) )
      goto LABEL_9;
    if ( !dword_4F077C0 )
      goto LABEL_15;
  }
  else if ( !dword_4F077C0 )
  {
LABEL_15:
    if ( !dword_4F077BC || (_DWORD)qword_4F077B4 || qword_4F077A8 <= 0xEA5Fu )
    {
      if ( (*(_BYTE *)(a3 + 40) & 0x20) == 0 )
      {
        v11 = sub_6E1A20(a1);
        sub_6851C0(dword_4F077C0 == 0 ? 1049 : 1902, v11);
        return 0;
      }
LABEL_19:
      *(_BYTE *)(a3 + 41) |= 2u;
      return 0;
    }
    goto LABEL_5;
  }
  if ( (*(_BYTE *)(a3 + 40) & 2) == 0 )
    goto LABEL_15;
LABEL_5:
  v4 = sub_8D40F0(*(_QWORD *)(a2 + 120));
  for ( i = *(_BYTE *)(v4 + 140); i == 12; i = *(_BYTE *)(v4 + 140) )
    v4 = *(_QWORD *)(v4 + 160);
  if ( dword_4F077C4 == 2 )
  {
    if ( (unsigned __int8)(i - 9) <= 2u )
    {
      v13 = *(_QWORD *)(*(_QWORD *)v4 + 96LL);
      if ( *(_QWORD *)(v13 + 24) )
      {
        if ( (*(_BYTE *)(v13 + 177) & 2) == 0 )
        {
          if ( (*(_BYTE *)(a3 + 40) & 0x20) == 0 )
          {
            v14 = sub_6E1A20(a1);
            sub_6851C0(1356, v14);
            return 0;
          }
          goto LABEL_19;
        }
      }
    }
    if ( dword_4F077BC && (unsigned int)sub_6E1AD0(a1, v16) )
    {
      if ( (*(_BYTE *)(a3 + 40) & 0x20) == 0 )
      {
        v15 = sub_6E1A20(a1);
        sub_6851C0(3239, v15);
        return 0;
      }
      goto LABEL_19;
    }
  }
  else if ( dword_4F077C0 && *(char *)(a3 + 41) < 0 )
  {
    if ( (*(_BYTE *)(a3 + 40) & 0x20) == 0 )
    {
      v12 = sub_6E1A20(a1);
      sub_6851C0(1357, v12);
      return 0;
    }
    goto LABEL_19;
  }
LABEL_9:
  v6 = *(__int64 **)(a3 + 16);
  result = 1;
  if ( v6 )
  {
    v8 = *v6;
    if ( v8 )
    {
      v9 = *(_BYTE *)(v8 + 80);
      if ( v9 == 9 || v9 == 7 )
      {
        v10 = *(_QWORD *)(v8 + 88);
      }
      else
      {
        if ( v9 != 21 )
          return result;
        v10 = *(_QWORD *)(*(_QWORD *)(v8 + 88) + 192LL);
      }
      result = 1;
      if ( v10 )
        *(_BYTE *)(v10 + 174) |= 0x80u;
    }
  }
  return result;
}
