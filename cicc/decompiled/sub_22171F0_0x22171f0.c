// Function: sub_22171F0
// Address: 0x22171f0
//
unsigned __int64 __fastcall sub_22171F0(__int64 a1, unsigned __int64 a2, unsigned __int64 a3)
{
  unsigned __int64 v4; // rbx

  if ( a2 < a3 )
  {
    v4 = a2;
    do
    {
      v4 += 4LL;
      *(_DWORD *)(v4 - 4) = __towupper_l();
    }
    while ( a3 > v4 );
  }
  return a3;
}
