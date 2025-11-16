// Function: sub_A212B0
// Address: 0xa212b0
//
void __fastcall sub_A212B0(__int64 a1, unsigned int a2, __int64 *a3, unsigned int a4)
{
  __int64 v5; // r15
  __int64 v6; // rbx
  unsigned int v7; // esi

  if ( a4 )
  {
    sub_A20C50(a1, a4, *a3, a3[1], 0, 0, a2, 1);
  }
  else
  {
    v5 = a3[1];
    v6 = 0;
    sub_A17B10(a1, 3u, *(_DWORD *)(a1 + 56));
    sub_A17CC0(a1, a2, 6);
    sub_A17CC0(a1, v5, 6);
    if ( (_DWORD)v5 )
    {
      do
      {
        v7 = *(_DWORD *)(*a3 + v6);
        v6 += 4;
        sub_A17CC0(a1, v7, 6);
      }
      while ( v6 != 4LL * (unsigned int)v5 );
    }
  }
}
