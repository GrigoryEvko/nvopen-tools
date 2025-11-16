// Function: sub_2957E60
// Address: 0x2957e60
//
bool __fastcall sub_2957E60(unsigned __int8 *a1)
{
  __int64 v2; // rdi
  bool result; // al
  __int64 v4; // r8
  __int64 v5; // rdx
  __int64 v6; // rcx
  _BYTE *v7; // rdi

  if ( *a1 <= 0x1Cu )
    return 0;
  v2 = *((_QWORD *)a1 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v2 + 8) - 17 <= 1 )
    v2 = **(_QWORD **)(v2 + 16);
  result = sub_BCAC40(v2, 1);
  if ( result )
  {
    v5 = *a1;
    if ( (_BYTE)v5 == 58 )
      return result;
    if ( (_BYTE)v5 == 86 )
    {
      v6 = *((_QWORD *)a1 + 1);
      if ( *(_QWORD *)(*((_QWORD *)a1 - 12) + 8LL) == v6 )
      {
        v7 = (_BYTE *)*((_QWORD *)a1 - 8);
        if ( *v7 <= 0x15u )
          return sub_AD7A80(v7, 1, v5, v6, v4);
      }
    }
  }
  return 0;
}
