// Function: sub_F4FC40
// Address: 0xf4fc40
//
__int64 __fastcall sub_F4FC40(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  __int64 v4; // rsi
  _QWORD *v5; // rax
  _QWORD *v6; // rdx
  unsigned int v8; // r8d

  v2 = *(_QWORD *)(a2 + 24);
  v3 = 0;
  if ( *(_BYTE *)v2 <= 0x1Cu )
    return v3;
  v4 = *(_QWORD *)(v2 + 40);
  if ( v4 == *(_QWORD *)(a1 + 160) )
    return v3;
  v3 = *(unsigned __int8 *)(a1 + 28);
  if ( (_BYTE)v3 )
  {
    v5 = *(_QWORD **)(a1 + 8);
    v6 = &v5[*(unsigned int *)(a1 + 20)];
    if ( v5 == v6 )
      return 0;
    while ( v4 != *v5 )
    {
      if ( v6 == ++v5 )
        return 0;
    }
    return v3;
  }
  LOBYTE(v8) = sub_C8CA60(a1, v4) != 0;
  return v8;
}
