// Function: sub_B8A870
// Address: 0xb8a870
//
__int64 __fastcall sub_B8A870(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  unsigned int v4; // r12d
  unsigned __int64 v5; // rsi
  int v6; // eax

  v2 = a2 + 24;
  v3 = *(_QWORD *)(a2 + 32);
  if ( v3 == a2 + 24 )
  {
    return 0;
  }
  else
  {
    v4 = 0;
    do
    {
      v5 = v3 - 56;
      if ( !v3 )
        v5 = 0;
      v6 = sub_B89FF0(a1, v5);
      v3 = *(_QWORD *)(v3 + 8);
      v4 |= v6;
    }
    while ( v2 != v3 );
  }
  return v4;
}
