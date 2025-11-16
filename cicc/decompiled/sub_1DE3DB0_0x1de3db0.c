// Function: sub_1DE3DB0
// Address: 0x1de3db0
//
void __fastcall sub_1DE3DB0(__int64 *a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // r14
  __int64 v5; // r10
  __int64 v6; // r11
  __int64 v7; // r15
  __int64 v8; // rcx

  v3 = (__int64)a3 + a2 - (_QWORD)a1;
  if ( a2 - (__int64)a1 <= 96 )
  {
    sub_1DE3CF0((__int64)a1, a2);
  }
  else
  {
    v5 = (__int64)a1;
    do
      sub_1DE3CF0(v5, v5 + 112);
    while ( a2 - v5 > 96 );
    sub_1DE3CF0(v5, a2);
    if ( v6 > 112 )
    {
      v7 = 7;
      do
      {
        sub_1DE3820(a1, a2, (__int64)a3, v7);
        v8 = 2 * v7;
        v7 *= 4;
        sub_1DE3820(a3, v3, (__int64)a1, v8);
      }
      while ( (a2 - (__int64)a1) >> 4 > v7 );
    }
  }
}
