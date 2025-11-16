// Function: sub_173FB90
// Address: 0x173fb90
//
__int64 __fastcall sub_173FB90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  _QWORD *v6; // rdx
  _BYTE *v7; // r13
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // eax
  unsigned int v11; // r14d
  bool v12; // al
  __int64 v13; // rax
  unsigned int v14; // r13d
  int v15; // r14d
  unsigned int v16; // r15d
  __int64 v17; // rax
  char v18; // dl
  unsigned int v19; // edx

  if ( *(_BYTE *)(a2 + 16) != 75 )
    return 0;
  v5 = *(_QWORD *)(a2 - 48);
  if ( *(_BYTE *)(v5 + 16) <= 0x17u )
    return 0;
  v6 = *(_QWORD **)(a1 + 8);
  *v6 = v5;
  v7 = *(_BYTE **)(a2 - 24);
  if ( v7[16] > 0x10u )
    return 0;
  if ( sub_1593BB0(*(_QWORD *)(a2 - 24), a2, (__int64)v6, a4) )
  {
LABEL_6:
    v10 = *(unsigned __int16 *)(a2 + 18);
    BYTE1(v10) &= ~0x80u;
    **(_DWORD **)a1 = v10;
    return 1;
  }
  if ( v7[16] == 13 )
  {
    v11 = *((_DWORD *)v7 + 8);
    if ( v11 <= 0x40 )
      v12 = *((_QWORD *)v7 + 3) == 0;
    else
      v12 = v11 == (unsigned int)sub_16A57B0((__int64)(v7 + 24));
  }
  else
  {
    if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) != 16 )
      return 0;
    v13 = sub_15A1020(v7, a2, v8, v9);
    if ( !v13 || *(_BYTE *)(v13 + 16) != 13 )
    {
      v15 = *(_QWORD *)(*(_QWORD *)v7 + 32LL);
      if ( !v15 )
        goto LABEL_6;
      v16 = 0;
      while ( 1 )
      {
        v17 = sub_15A0A60((__int64)v7, v16);
        if ( !v17 )
          return 0;
        v18 = *(_BYTE *)(v17 + 16);
        if ( v18 != 9 )
        {
          if ( v18 != 13 )
            return 0;
          v19 = *(_DWORD *)(v17 + 32);
          if ( v19 <= 0x40 )
          {
            if ( *(_QWORD *)(v17 + 24) )
              return 0;
          }
          else if ( v19 != (unsigned int)sub_16A57B0(v17 + 24) )
          {
            return 0;
          }
        }
        if ( v15 == ++v16 )
          goto LABEL_6;
      }
    }
    v14 = *(_DWORD *)(v13 + 32);
    if ( v14 <= 0x40 )
      v12 = *(_QWORD *)(v13 + 24) == 0;
    else
      v12 = v14 == (unsigned int)sub_16A57B0(v13 + 24);
  }
  if ( v12 )
    goto LABEL_6;
  return 0;
}
