// Function: sub_74A260
// Address: 0x74a260
//
_DWORD *__fastcall sub_74A260(__int64 a1, void (__fastcall **a2)(char *))
{
  _DWORD *result; // rax

  if ( (*(_BYTE *)(a1 + 143) & 2) != 0 )
    (*a2)(" __attribute((__may_alias__))");
  result = &dword_4F077C4;
  if ( dword_4F077C4 != 2 )
  {
    result = &unk_4F07778;
    if ( unk_4F07778 > 201111 )
      return (_DWORD *)sub_74A1C0(*(__int64 **)(a1 + 104), 1u, a2);
  }
  return result;
}
