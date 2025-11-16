// Function: sub_117C420
// Address: 0x117c420
//
bool __fastcall sub_117C420(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  bool result; // al
  __int64 v5; // r8
  __int64 v6; // rdx
  __int64 v7; // rcx
  _BYTE *v8; // rdi

  v2 = *(_QWORD *)(a1 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v2 + 8) - 17 <= 1 )
    v2 = **(_QWORD **)(v2 + 16);
  if ( sub_BCAC40(v2, 1) )
  {
    if ( *(_BYTE *)a1 == 57 )
      return 1;
    v3 = *(_QWORD *)(a1 + 8);
    if ( *(_BYTE *)a1 != 86 || *(_QWORD *)(*(_QWORD *)(a1 - 96) + 8LL) != v3 || **(_BYTE **)(a1 - 32) > 0x15u )
      goto LABEL_10;
    if ( sub_AC30F0(*(_QWORD *)(a1 - 32)) )
      return 1;
  }
  v3 = *(_QWORD *)(a1 + 8);
LABEL_10:
  if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
    v3 = **(_QWORD **)(v3 + 16);
  result = sub_BCAC40(v3, 1);
  if ( result )
  {
    if ( *(_BYTE *)a1 != 58 )
    {
      if ( *(_BYTE *)a1 != 86 )
        return 0;
      v6 = *(_QWORD *)(a1 - 96);
      v7 = *(_QWORD *)(a1 + 8);
      result = 0;
      if ( *(_QWORD *)(v6 + 8) == v7 )
      {
        v8 = *(_BYTE **)(a1 - 64);
        if ( *v8 <= 0x15u )
          return sub_AD7A80(v8, 1, v6, v7, v5);
      }
      return result;
    }
    return 1;
  }
  return result;
}
