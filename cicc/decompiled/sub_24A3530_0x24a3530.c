// Function: sub_24A3530
// Address: 0x24a3530
//
bool __fastcall sub_24A3530(__int64 a1)
{
  unsigned int v1; // r12d
  __int64 i; // r14
  unsigned __int64 v3; // rbx
  int v4; // eax
  __int64 v5; // rbx
  int v6; // r13d
  unsigned int j; // r15d

  v1 = 0;
  for ( i = *(_QWORD *)(a1 + 80); a1 + 72 != i; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    v3 = *(_QWORD *)(i + 24) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v3 == i + 24 )
    {
      v5 = 0;
    }
    else
    {
      if ( !v3 )
        BUG();
      v4 = *(unsigned __int8 *)(v3 - 24);
      v5 = v3 - 24;
      if ( (unsigned int)(v4 - 30) >= 0xB )
        v5 = 0;
    }
    v6 = sub_B46E30(v5);
    if ( v6 )
    {
      for ( j = 0; j != v6; ++j )
        v1 -= ((unsigned __int8)sub_D0E970(v5, j, 0) == 0) - 1;
    }
  }
  return (unsigned int)qword_4FEAC08 < v1;
}
