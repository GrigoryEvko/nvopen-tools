// Function: sub_31B84B0
// Address: 0x31b84b0
//
__int64 __fastcall sub_31B84B0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v3; // rax
  int v4; // eax
  bool v5; // al
  unsigned __int8 *v6; // rdx
  int v7; // eax
  unsigned __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rax
  int v12; // edx

  v1 = *(_QWORD *)(a1 + 16);
  if ( (unsigned __int8)sub_B46420(v1) || (unsigned __int8)sub_B46490(v1) )
  {
    if ( *(_BYTE *)v1 != 85 )
      return 1;
    v3 = *(_QWORD *)(v1 - 32);
    if ( !v3 )
      return 1;
    if ( *(_BYTE *)v3 )
      return 1;
    if ( *(_QWORD *)(v3 + 24) != *(_QWORD *)(v1 + 80) )
      return 1;
    if ( (*(_BYTE *)(v3 + 33) & 0x20) == 0 )
      return 1;
    v4 = *(_DWORD *)(v3 + 36);
    if ( v4 != 324 && v4 != 291 )
      return 1;
  }
  v5 = sub_318B630(a1);
  v6 = *(unsigned __int8 **)(a1 + 16);
  if ( v5 && *(_DWORD *)(a1 + 8) == 54 && (v6[2] & 0x40) != 0 )
    return 1;
  v7 = *v6;
  if ( (_BYTE)v7 != 85 )
  {
    v8 = (unsigned int)(v7 - 29);
    if ( (unsigned int)v8 > 0x38 )
      return 0;
    v9 = 0x110000800000220LL;
    return _bittest64(&v9, v8) != 0;
  }
  v10 = *((_QWORD *)v6 - 4);
  if ( !v10
    || !*(_BYTE *)v10
    && *(_QWORD *)(v10 + 24) == *((_QWORD *)v6 + 10)
    && (*(_BYTE *)(v10 + 33) & 0x20) != 0
    && (unsigned int)(*(_DWORD *)(v10 + 36) - 342) <= 1 )
  {
    return 1;
  }
  if ( *(_BYTE *)v10 )
    return 1;
  v11 = *((_QWORD *)v6 + 10);
  if ( *(_QWORD *)(v10 + 24) != v11 || (*(_BYTE *)(v10 + 33) & 0x20) == 0 )
    return 1;
  v12 = *(_DWORD *)(v10 + 36);
  LOBYTE(v11) = v12 != 324;
  LOBYTE(v12) = v12 != 291;
  return v12 & (unsigned int)v11;
}
