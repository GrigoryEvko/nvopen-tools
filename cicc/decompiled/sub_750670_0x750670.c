// Function: sub_750670
// Address: 0x750670
//
__int64 __fastcall sub_750670(__int64 a1, char a2)
{
  __int64 result; // rax
  __int64 v3; // rdx

  *(_BYTE *)(a1 - 8) &= ~0x80u;
  if ( a2 == 6 )
  {
    result = (unsigned int)*(unsigned __int8 *)(a1 + 140) - 9;
    if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) <= 2u && *(char *)(a1 + 178) < 0 )
    {
      result = *(_QWORD *)(a1 + 168);
      v3 = *(_QWORD *)(result + 152);
      *(_BYTE *)(result - 8) &= ~0x80u;
      if ( v3 )
      {
        if ( (*(_BYTE *)(v3 + 29) & 0x20) == 0 )
          *(_BYTE *)(v3 - 8) &= ~0x80u;
      }
    }
  }
  else if ( a2 == 28 && (*(_BYTE *)(a1 + 124) & 1) == 0 )
  {
    result = *(_QWORD *)(a1 + 128);
    *(_BYTE *)(result - 8) &= ~0x80u;
  }
  return result;
}
