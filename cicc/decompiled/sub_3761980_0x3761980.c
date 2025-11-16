// Function: sub_3761980
// Address: 0x3761980
//
__int64 __fastcall sub_3761980(__int64 a1, unsigned __int64 a2, unsigned int a3)
{
  int v3; // r13d
  __int64 v4; // r12
  unsigned int i; // ebx

  v3 = *(_DWORD *)(a2 + 68);
  if ( v3 )
  {
    v4 = 0;
    for ( i = 0; i != v3; ++i )
    {
      if ( a3 != i )
        sub_3760E70(a1, a2, i, *(_QWORD *)(v4 + *(_QWORD *)(a2 + 40)), *(_QWORD *)(v4 + *(_QWORD *)(a2 + 40) + 8));
      v4 += 40;
    }
  }
  return *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL * a3);
}
