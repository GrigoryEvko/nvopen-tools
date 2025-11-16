// Function: sub_225FE40
// Address: 0x225fe40
//
__int64 __fastcall sub_225FE40(__int64 a1)
{
  unsigned int v1; // r14d
  __int64 v2; // rbx
  int v3; // r12d
  unsigned __int64 v4; // rax
  unsigned int v5; // r12d

  v1 = 0;
  v2 = *(_QWORD *)(a1 + 80);
  if ( v2 != a1 + 72 )
  {
    v3 = 0;
    do
    {
      if ( !v2 )
        BUG();
      ++v1;
      v4 = *(_QWORD *)(v2 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v4 != v2 + 24 )
      {
        if ( !v4 )
          BUG();
        if ( (unsigned int)*(unsigned __int8 *)(v4 - 24) - 30 <= 0xA )
          v3 += sub_B46E30(v4 - 24);
      }
      v2 = *(_QWORD *)(v2 + 8);
    }
    while ( a1 + 72 != v2 );
    v5 = v3 + 2 - v1;
    if ( v1 )
      return v5;
  }
  return v1;
}
