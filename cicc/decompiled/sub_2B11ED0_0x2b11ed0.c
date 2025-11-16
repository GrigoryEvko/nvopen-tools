// Function: sub_2B11ED0
// Address: 0x2b11ed0
//
void __fastcall sub_2B11ED0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r14
  __int64 v6; // r15
  __int64 v7; // rdi
  __int64 v8; // r15
  __int64 v9; // rcx
  __int64 v10; // [rsp+0h] [rbp-40h]
  __int64 v11; // [rsp+8h] [rbp-38h]

  v5 = a3 + a2 - a1;
  v10 = a2 - a1;
  v11 = (a2 - a1) >> 4;
  if ( a2 - a1 <= 96 )
  {
    sub_2B0F290(a1, a2);
  }
  else
  {
    v6 = a1;
    do
    {
      v7 = v6;
      v6 += 112;
      sub_2B0F290(v7, v6);
    }
    while ( a2 - v6 > 96 );
    sub_2B0F290(v6, a2);
    if ( v10 > 112 )
    {
      v8 = 7;
      do
      {
        sub_2B11E30(a1, a2, a3, v8);
        v9 = 2 * v8;
        v8 *= 4;
        sub_2B11E30(a3, v5, a1, v9);
      }
      while ( v11 > v8 );
    }
  }
}
