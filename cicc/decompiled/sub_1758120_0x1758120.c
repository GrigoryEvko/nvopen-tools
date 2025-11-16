// Function: sub_1758120
// Address: 0x1758120
//
char __fastcall sub_1758120(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rdx
  __int64 v5; // rcx
  unsigned int v7; // r12d
  __int64 v8; // rax
  unsigned int v9; // ebx
  int v10; // r12d
  unsigned int v11; // r13d
  __int64 v12; // rax
  char v13; // dl
  unsigned int v14; // r14d

  if ( a1[16] > 0x10u )
    return 0;
  if ( sub_1593BB0((__int64)a1, a2, a3, a4) )
    return 1;
  if ( a1[16] == 13 )
  {
    v7 = *((_DWORD *)a1 + 8);
    if ( v7 <= 0x40 )
      return *((_QWORD *)a1 + 3) == 0;
    else
      return v7 == (unsigned int)sub_16A57B0((__int64)(a1 + 24));
  }
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 16 )
    return 0;
  v8 = sub_15A1020(a1, a2, v4, v5);
  if ( !v8 || *(_BYTE *)(v8 + 16) != 13 )
  {
    v10 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
    if ( !v10 )
      return 1;
    v11 = 0;
    while ( 1 )
    {
      v12 = sub_15A0A60((__int64)a1, v11);
      if ( !v12 )
        break;
      v13 = *(_BYTE *)(v12 + 16);
      if ( v13 != 9 )
      {
        if ( v13 != 13 )
          return 0;
        v14 = *(_DWORD *)(v12 + 32);
        if ( v14 <= 0x40 )
        {
          if ( *(_QWORD *)(v12 + 24) )
            return 0;
        }
        else if ( v14 != (unsigned int)sub_16A57B0(v12 + 24) )
        {
          return 0;
        }
      }
      if ( v10 == ++v11 )
        return 1;
    }
    return 0;
  }
  v9 = *(_DWORD *)(v8 + 32);
  if ( v9 <= 0x40 )
    return *(_QWORD *)(v8 + 24) == 0;
  else
    return v9 == (unsigned int)sub_16A57B0(v8 + 24);
}
