// Function: sub_1719260
// Address: 0x1719260
//
__int64 __fastcall sub_1719260(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r12d
  unsigned __int8 v5; // al
  __int64 v7; // rax
  unsigned int v8; // ebx
  unsigned int v9; // r14d
  int v10; // r13d
  __int64 v11; // rax
  char v12; // dl
  unsigned int v13; // r15d

  v5 = a1[16];
  if ( v5 == 13 )
  {
    v4 = *((_DWORD *)a1 + 8);
    if ( v4 <= 0x40 )
      LOBYTE(v4) = *((_QWORD *)a1 + 3) == 0;
    else
      LOBYTE(v4) = v4 == (unsigned int)sub_16A57B0((__int64)(a1 + 24));
    return v4;
  }
  LOBYTE(v4) = v5 <= 0x10u && *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16;
  if ( !(_BYTE)v4 )
    return 0;
  v7 = sub_15A1020(a1, a2, *(_QWORD *)a1, a4);
  if ( !v7 || *(_BYTE *)(v7 + 16) != 13 )
  {
    v9 = 0;
    v10 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
    if ( !v10 )
      return v4;
    while ( 1 )
    {
      v11 = sub_15A0A60((__int64)a1, v9);
      if ( !v11 )
        break;
      v12 = *(_BYTE *)(v11 + 16);
      if ( v12 != 9 )
      {
        if ( v12 != 13 )
          return 0;
        v13 = *(_DWORD *)(v11 + 32);
        if ( v13 <= 0x40 )
        {
          if ( *(_QWORD *)(v11 + 24) )
            return 0;
        }
        else if ( v13 != (unsigned int)sub_16A57B0(v11 + 24) )
        {
          return 0;
        }
      }
      if ( v10 == ++v9 )
        return v4;
    }
    return 0;
  }
  v8 = *(_DWORD *)(v7 + 32);
  if ( v8 <= 0x40 )
    LOBYTE(v4) = *(_QWORD *)(v7 + 24) == 0;
  else
    LOBYTE(v4) = v8 == (unsigned int)sub_16A57B0(v7 + 24);
  return v4;
}
