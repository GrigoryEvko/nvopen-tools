// Function: sub_2013D30
// Address: 0x2013d30
//
__int64 __fastcall sub_2013D30(
        __int64 a1,
        unsigned __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        const __m128i *a6)
{
  int v6; // r13d
  __int64 v7; // r12
  unsigned int i; // ebx

  v6 = *(_DWORD *)(a2 + 60);
  if ( v6 )
  {
    v7 = 0;
    for ( i = 0; i != v6; ++i )
    {
      if ( a3 != i )
        sub_2013400(a1, a2, i, *(_QWORD *)(v7 + *(_QWORD *)(a2 + 32)), *(__m128i **)(v7 + *(_QWORD *)(a2 + 32) + 8), a6);
      v7 += 40;
    }
  }
  return *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL * a3);
}
