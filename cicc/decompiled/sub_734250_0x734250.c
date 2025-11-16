// Function: sub_734250
// Address: 0x734250
//
void __fastcall sub_734250(__int64 a1, unsigned int a2)
{
  __int64 v2; // rax

  if ( (*(_BYTE *)(a1 + 51) & 0x40) != 0 )
  {
    v2 = sub_730770(a1, 0);
    sub_734250(v2, a2);
  }
  if ( *(_QWORD *)(a1 + 16) )
  {
    *(_BYTE *)(a1 + 50) |= 1u;
    if ( a2 )
      sub_7340D0(a1, 0, 0);
  }
}
