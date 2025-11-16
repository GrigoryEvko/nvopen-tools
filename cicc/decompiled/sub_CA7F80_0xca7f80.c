// Function: sub_CA7F80
// Address: 0xca7f80
//
unsigned __int64 __fastcall sub_CA7F80(__int64 a1, _BYTE *a2)
{
  unsigned __int64 result; // rax

  if ( *(_BYTE **)(a1 + 48) == a2 )
    return 0;
  result = 0;
  if ( *a2 <= 0x20u )
    return (0x100002600uLL >> *a2) & 1;
  return result;
}
