// Function: sub_80AA60
// Address: 0x80aa60
//
_BOOL8 __fastcall sub_80AA60(__int64 a1)
{
  char v1; // al
  __int64 v2; // rdx
  _BOOL8 result; // rax

  v1 = *(_BYTE *)(a1 + 89);
  if ( (v1 & 0x40) != 0 || ((v1 & 8) != 0 ? (v2 = *(_QWORD *)(a1 + 24)) : (v2 = *(_QWORD *)(a1 + 8)), result = 0, !v2) )
  {
    result = 0;
    if ( *(_QWORD *)a1 )
    {
      result = 1;
      if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) <= 2u )
        return *(_BYTE *)(*(_QWORD *)(a1 + 168) + 113LL) == 0;
    }
  }
  return result;
}
