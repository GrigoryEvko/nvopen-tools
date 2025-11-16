// Function: sub_173DBE0
// Address: 0x173dbe0
//
_BOOL8 __fastcall sub_173DBE0(__int64 a1)
{
  _BOOL8 result; // rax
  __int64 v2; // rdx

  result = 0;
  if ( *(_BYTE *)(a1 + 16) == 78 )
  {
    v2 = *(_QWORD *)(a1 - 24);
    if ( !*(_BYTE *)(v2 + 16) )
      return (*(_BYTE *)(v2 + 33) & 0x20) != 0;
  }
  return result;
}
