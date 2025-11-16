// Function: sub_164F8A0
// Address: 0x164f8a0
//
__int64 __fastcall sub_164F8A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned int v3; // eax
  __int64 v4; // r13
  unsigned int v5; // r8d
  unsigned int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rdi
  unsigned int v10; // r15d

  v2 = a1;
  v3 = *(_DWORD *)(a1 + 8);
  while ( v3 - 2 <= 1 )
  {
    v4 = v3;
    if ( **(_BYTE **)(v2 - 8LL * v3) )
      break;
    if ( v3 == 3 )
    {
      v8 = *(_QWORD *)(v2 - 8);
      if ( *(_BYTE *)v8 != 1 )
        return 0;
      v9 = *(_QWORD *)(v8 + 136);
      if ( *(_BYTE *)(v9 + 16) != 13 )
        return 0;
      v10 = *(_DWORD *)(v9 + 32);
      if ( v10 <= 0x40 )
      {
        if ( *(_QWORD *)(v9 + 24) )
          return 0;
      }
      else if ( v10 != (unsigned int)sub_16A57B0(v9 + 24) )
      {
        return 0;
      }
    }
    v2 = *(_QWORD *)(v2 + 8 * (1 - v4));
    if ( !v2 )
      return 0;
    if ( (unsigned __int8)(*(_BYTE *)v2 - 4) > 0x1Eu )
      return 0;
    sub_1412190(a2, v2);
    v5 = v7;
    if ( !(_BYTE)v7 )
      return 0;
    v3 = *(_DWORD *)(v2 + 8);
    if ( v3 <= 1 )
      return v5;
  }
  return 0;
}
