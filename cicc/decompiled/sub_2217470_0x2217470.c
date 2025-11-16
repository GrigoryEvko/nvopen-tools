// Function: sub_2217470
// Address: 0x2217470
//
unsigned __int64 __fastcall sub_2217470(__int64 a1, unsigned __int64 a2, unsigned __int64 a3, __int64 a4)
{
  __int64 v4; // rax

  if ( a2 < a3 )
  {
    v4 = 0;
    do
    {
      *(_DWORD *)(a4 + 4 * v4) = *(_DWORD *)(a1 + 4LL * *(unsigned __int8 *)(a2 + v4) + 156);
      ++v4;
    }
    while ( a3 - a2 != v4 );
  }
  return a3;
}
