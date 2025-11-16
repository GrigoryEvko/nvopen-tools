// Function: sub_3540EB0
// Address: 0x3540eb0
//
void __fastcall sub_3540EB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rsi
  __int64 v7; // r15
  __int64 v8; // r14
  __int64 v9; // rdi
  __int64 v10; // r14
  __int64 v11; // rcx

  v4 = a2 - a1;
  v7 = a3 + v4;
  if ( v4 <= 528 )
  {
    sub_35407B0(a1, a2);
  }
  else
  {
    v8 = a1;
    do
    {
      v9 = v8;
      v8 += 616;
      sub_35407B0(v9, v8);
    }
    while ( a2 - v8 > 528 );
    sub_35407B0(v8, a2);
    if ( v4 > 616 )
    {
      v10 = 7;
      do
      {
        sub_35406E0(a1, a2, a3, v10);
        v11 = 2 * v10;
        v10 *= 4;
        sub_35406E0(a3, v7, a1, v11);
      }
      while ( 0x2E8BA2E8BA2E8BA3LL * (v4 >> 3) > v10 );
    }
  }
}
