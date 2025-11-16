// Function: sub_2D43010
// Address: 0x2d43010
//
__int64 __fastcall sub_2D43010(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int8 v3; // dl

  result = 0;
  if ( (*(_WORD *)(a2 + 2) & 0x1F0) == 0 )
  {
    v3 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL) + 8LL);
    result = 1;
    if ( v3 > 3u && v3 != 5 )
      return (v3 == 14) | (unsigned __int8)((v3 & 0xFD) == 4);
  }
  return result;
}
