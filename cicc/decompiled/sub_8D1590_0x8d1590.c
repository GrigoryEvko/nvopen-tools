// Function: sub_8D1590
// Address: 0x8d1590
//
__int64 __fastcall sub_8D1590(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // rdx
  unsigned int v4; // r8d
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rdi
  __int64 v11; // rsi

  v2 = *(_BYTE *)(a1 + 169);
  if ( (v2 & 2) != 0 )
    return 0;
  v3 = *(unsigned __int8 *)(a2 + 169);
  if ( (v3 & 2) != 0 )
    return 0;
  v6 = v3 & 1;
  if ( (v2 & 1) != 0 )
  {
    if ( (_BYTE)v6 )
    {
      v7 = *(_QWORD *)(a1 + 176);
      v4 = 0;
      if ( *(_BYTE *)(v7 + 24) != 2 )
        return v4;
      v8 = *(_QWORD *)(a2 + 176);
      if ( *(_BYTE *)(v8 + 24) != 2 )
        return v4;
      return sub_73A2C0(*(_QWORD *)(v7 + 56), *(_QWORD *)(v8 + 56), v8, v6, 0);
    }
    return 0;
  }
  if ( (_BYTE)v6 )
    return 0;
  v4 = 0;
  v9 = *(unsigned __int8 *)(a2 + 168);
  if ( *(char *)(a1 + 168) >= 0 )
  {
    if ( (v9 & 0x80u) == 0LL && *(_QWORD *)(a1 + 176) == *(_QWORD *)(a2 + 176) )
      return (((unsigned __int8)(v3 ^ v2) >> 5) ^ 1) & 1;
    return v4;
  }
  if ( (v9 & 0x80u) == 0LL )
    return v4;
  v10 = *(_QWORD *)(a1 + 176);
  v11 = *(_QWORD *)(a2 + 176);
  if ( !v10 || !v11 )
    return v10 == v11;
  return sub_73A2C0(v10, v11, v3, v9, 0);
}
