// Function: sub_A23520
// Address: 0xa23520
//
void __fastcall sub_A23520(__int64 a1, unsigned int a2, __int64 a3, unsigned int a4)
{
  unsigned int v5; // r15d
  __int64 v6; // rbx
  unsigned int v7; // esi

  v5 = *(_DWORD *)(a3 + 8);
  if ( a4 )
  {
    sub_A20C50(a1, a4, *(_QWORD *)a3, v5, 0, 0, a2, 1);
  }
  else
  {
    v6 = 0;
    sub_A17B10(a1, 3u, *(_DWORD *)(a1 + 56));
    sub_A17CC0(a1, a2, 6);
    sub_A17CC0(a1, v5, 6);
    if ( v5 )
    {
      do
      {
        v7 = *(_DWORD *)(*(_QWORD *)a3 + v6);
        v6 += 4;
        sub_A17CC0(a1, v7, 6);
      }
      while ( 4LL * v5 != v6 );
    }
  }
}
