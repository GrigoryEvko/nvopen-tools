// Function: sub_91C310
// Address: 0x91c310
//
__int64 __fastcall sub_91C310(__int64 a1)
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
        return sub_91C2E0(*(_QWORD *)(a1 + 120));
    }
  }
  return result;
}
