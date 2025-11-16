// Function: sub_127BFC0
// Address: 0x127bfc0
//
__int64 __fastcall sub_127BFC0(__int64 a1)
{
  char v1; // dl
  __int64 result; // rax

  v1 = *(_BYTE *)(a1 + 156);
  result = 3;
  if ( (v1 & 2) == 0 )
  {
    result = 4;
    if ( (v1 & 4) == 0 )
    {
      result = 1;
      if ( (*(_BYTE *)(a1 + 156) & 1) == 0 )
        return sub_127BF90(*(_QWORD *)(a1 + 120));
    }
  }
  return result;
}
