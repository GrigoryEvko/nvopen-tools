// Function: sub_13CD570
// Address: 0x13cd570
//
_QWORD *__fastcall sub_13CD570(_QWORD *a1, __int64 a2)
{
  int v2; // eax
  _QWORD *result; // rax
  __int64 v4; // rax
  unsigned int v5; // r13d
  bool v6; // al
  int v7; // eax
  _QWORD **v8; // rbx
  unsigned int v9; // r13d
  bool v10; // al
  int v11; // r13d
  unsigned int v12; // r14d
  __int64 v13; // rax
  char v14; // dl
  unsigned int v15; // r15d

  v2 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)v2 <= 0x10u )
  {
    if ( (unsigned __int8)sub_1593BB0(a2) )
      return (_QWORD *)sub_15A06D0(*a1);
    v2 = *(unsigned __int8 *)(a2 + 16);
    if ( (_BYTE)v2 == 13 )
    {
      v9 = *(_DWORD *)(a2 + 32);
      if ( v9 <= 0x40 )
        v10 = *(_QWORD *)(a2 + 24) == 0;
      else
        v10 = v9 == (unsigned int)sub_16A57B0(a2 + 24);
      if ( v10 )
        return (_QWORD *)sub_15A06D0(*a1);
      return 0;
    }
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
    {
      v4 = sub_15A1020(a2);
      if ( v4 && *(_BYTE *)(v4 + 16) == 13 )
      {
        v5 = *(_DWORD *)(v4 + 32);
        if ( v5 <= 0x40 )
          v6 = *(_QWORD *)(v4 + 24) == 0;
        else
          v6 = v5 == (unsigned int)sub_16A57B0(v4 + 24);
        if ( v6 )
          return (_QWORD *)sub_15A06D0(*a1);
      }
      else
      {
        v11 = *(_QWORD *)(*(_QWORD *)a2 + 32LL);
        if ( !v11 )
          return (_QWORD *)sub_15A06D0(*a1);
        v12 = 0;
        while ( 1 )
        {
          v13 = sub_15A0A60(a2, v12);
          if ( !v13 )
            break;
          v14 = *(_BYTE *)(v13 + 16);
          if ( v14 != 9 )
          {
            if ( v14 != 13 )
              break;
            v15 = *(_DWORD *)(v13 + 32);
            if ( v15 <= 0x40 )
            {
              if ( *(_QWORD *)(v13 + 24) )
                break;
            }
            else if ( v15 != (unsigned int)sub_16A57B0(v13 + 24) )
            {
              break;
            }
          }
          if ( v11 == ++v12 )
            return (_QWORD *)sub_15A06D0(*a1);
        }
      }
      v2 = *(unsigned __int8 *)(a2 + 16);
    }
  }
  if ( (unsigned __int8)v2 > 0x17u )
  {
    v7 = v2 - 24;
  }
  else
  {
    if ( (_BYTE)v2 != 5 )
      return 0;
    v7 = *(unsigned __int16 *)(a2 + 18);
  }
  if ( v7 != 45 )
    return 0;
  v8 = (*(_BYTE *)(a2 + 23) & 0x40) != 0
     ? *(_QWORD ***)(a2 - 8)
     : (_QWORD **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  result = *v8;
  if ( !*v8 )
    return 0;
  if ( *result != *a1 )
    return 0;
  return result;
}
