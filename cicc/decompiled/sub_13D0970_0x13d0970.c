// Function: sub_13D0970
// Address: 0x13d0970
//
_BOOL8 __fastcall sub_13D0970(__int64 a1)
{
  __int64 v1; // rbx
  _BOOL8 result; // rax

  v1 = a1;
  if ( *(_QWORD *)(a1 + 8) == sub_16982C0() )
    v1 = *(_QWORD *)(a1 + 16);
  result = 0;
  if ( (*(_BYTE *)(v1 + 26) & 7) == 3 )
    return (*(_BYTE *)(v1 + 26) & 8) != 0;
  return result;
}
