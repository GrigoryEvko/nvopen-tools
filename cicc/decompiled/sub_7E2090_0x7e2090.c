// Function: sub_7E2090
// Address: 0x7e2090
//
_BOOL8 __fastcall sub_7E2090(__int64 a1)
{
  _BOOL8 result; // rax
  __int64 v2; // rdx

  result = 0;
  if ( *(_BYTE *)(a1 + 24) == 3 )
  {
    v2 = *(_QWORD *)(a1 + 56);
    if ( !*(_QWORD *)(v2 + 8) && *(_BYTE *)(v2 + 177) == 1 )
      return (unsigned int)sub_7E1F90(*(_QWORD *)(v2 + 120)) != 0;
  }
  return result;
}
