// Function: sub_731920
// Address: 0x731920
//
__int64 __fastcall sub_731920(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v6; // al
  __int64 v7; // r8
  char v8; // al
  char v9; // al
  unsigned __int8 v10; // dl
  __int64 v11; // r12
  __int64 result; // rax

  if ( !(_DWORD)a2 )
    return (unsigned int)sub_731770(a1, 0, a3, a4, a5, a6) == 0;
  while ( 1 )
  {
    v8 = *(_BYTE *)(a1 + 24);
    if ( v8 == 2 || v8 == 20 )
      return 1;
    if ( (*(_BYTE *)(a1 + 25) & 3) != 0 && !(_DWORD)a3 )
      break;
    if ( v8 == 3 )
      return *(_QWORD *)(*(_QWORD *)(a1 + 56) + 8LL) == 0;
    if ( v8 != 1 )
      return 0;
    v6 = *(_BYTE *)(a1 + 56);
    v7 = *(_QWORD *)(a1 + 72);
    if ( v6 )
    {
      if ( v6 != 21 )
        return 0;
    }
    if ( (*(_BYTE *)(v7 + 25) & 1) == 0 )
      return 0;
LABEL_8:
    a1 = v7;
    LODWORD(a3) = 0;
  }
  if ( v8 == 24 || v8 == 3 )
    return 1;
  if ( v8 != 1 )
    return 0;
  v9 = *(_BYTE *)(a1 + 56);
  v7 = *(_QWORD *)(a1 + 72);
  if ( (unsigned __int8)(v9 - 94) <= 1u || v9 == 3 )
    goto LABEL_8;
  v10 = v9 - 92;
  result = 0;
  if ( v10 <= 1u )
  {
    v11 = *(_QWORD *)(v7 + 16);
    result = sub_731920(*(_QWORD *)(a1 + 72), a2, 0);
    if ( (_DWORD)result )
      return (unsigned int)sub_731920(v11, (unsigned int)a2, 0) != 0;
  }
  return result;
}
