// Function: sub_AC56B0
// Address: 0xac56b0
//
__int64 __fastcall sub_AC56B0(__int64 a1)
{
  char v1; // al
  __int64 result; // rax

  v1 = *(_BYTE *)(a1 + 40);
  if ( (v1 & 1) != 0 )
    return (v1 & 2) != 0;
  *(_BYTE *)(a1 + 40) = v1 | 1;
  result = sub_AC5610(a1);
  *(_BYTE *)(a1 + 40) = (2 * (result & 1)) | *(_BYTE *)(a1 + 40) & 0xFD;
  return result;
}
