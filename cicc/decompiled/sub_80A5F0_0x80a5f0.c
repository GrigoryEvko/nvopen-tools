// Function: sub_80A5F0
// Address: 0x80a5f0
//
_BOOL8 __fastcall sub_80A5F0(__int64 a1)
{
  __int64 v1; // rdx
  _BOOL8 result; // rax

  v1 = *(_QWORD *)(a1 + 168);
  result = 0;
  if ( (*(_BYTE *)(v1 + 109) & 0x20) != 0 )
  {
    while ( *(_BYTE *)(a1 + 140) == 12 )
      a1 = *(_QWORD *)(a1 + 160);
    result = 0;
    if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 96LL) + 181LL) & 2) != 0 )
      return *(_QWORD *)(v1 + 240) != 0;
  }
  return result;
}
