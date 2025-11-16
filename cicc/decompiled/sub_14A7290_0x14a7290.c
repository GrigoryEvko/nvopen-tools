// Function: sub_14A7290
// Address: 0x14a7290
//
bool __fastcall sub_14A7290(__int64 a1)
{
  __int64 v1; // rdx
  char v2; // cl
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rax
  _BYTE *v9; // rdi
  __int64 v10; // rdx
  _BOOL4 v11; // eax

  v1 = *(unsigned int *)(a1 + 8);
  v2 = **(_BYTE **)(a1 - 8 * v1);
  if ( (unsigned __int8)(v2 - 4) <= 0x1Eu && (unsigned int)v1 > 2 )
  {
    v6 = *(_QWORD *)(a1 + 8 * (1 - v1));
    if ( !v6 || (unsigned __int8)(*(_BYTE *)v6 - 4) > 0x1Eu )
      BUG();
    v7 = *(unsigned int *)(v6 + 8);
    v8 = 0;
    if ( (unsigned int)v7 > 2 )
      v8 = 2LL * ((unsigned __int8)(**(_BYTE **)(v6 - 8 * v7) - 4) < 0x1Fu);
    v9 = *(_BYTE **)(v6 + 8 * (v8 - v7));
    if ( *v9 )
      return 0;
    v3 = sub_161E970(v9);
    if ( v10 != 14 )
      return 0;
  }
  else
  {
    if ( !*(_DWORD *)(a1 + 8) )
      return 0;
    if ( v2 )
      return 0;
    v3 = sub_161E970(*(_QWORD *)(a1 - 8 * v1));
    if ( v4 != 14 )
      return 0;
  }
  v11 = *(_QWORD *)v3 != 0x7020656C62617476LL || *(_DWORD *)(v3 + 8) != 1953393007 || *(_WORD *)(v3 + 12) != 29285;
  return !v11;
}
