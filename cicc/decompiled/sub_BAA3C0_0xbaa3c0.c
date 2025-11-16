// Function: sub_BAA3C0
// Address: 0xbaa3c0
//
__int64 __fastcall sub_BAA3C0(__int64 a1)
{
  __int64 v1; // r13
  unsigned int v2; // r12d
  __int64 v3; // rbx
  __int64 v4; // rdi
  int v5; // eax

  v1 = a1 + 24;
  v2 = 0;
  v3 = *(_QWORD *)(a1 + 32);
  if ( a1 + 24 != v3 )
  {
    do
    {
      v4 = v3 - 56;
      if ( !v3 )
        v4 = 0;
      v5 = sub_B2BED0(v4);
      v3 = *(_QWORD *)(v3 + 8);
      v2 += v5;
    }
    while ( v1 != v3 );
  }
  return v2;
}
