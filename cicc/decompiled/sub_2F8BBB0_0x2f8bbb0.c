// Function: sub_2F8BBB0
// Address: 0x2f8bbb0
//
void __fastcall sub_2F8BBB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx

  if ( a2 - a1 <= 1232 )
  {
    sub_2F8ADD0(a1, a2, a3, a4, a5, a6);
  }
  else
  {
    v6 = 88 * ((0x2E8BA2E8BA2E8BA3LL * ((a2 - a1) >> 3)) >> 1);
    sub_2F8BBB0(a1, a1 + v6);
    sub_2F8BBB0(a1 + v6, a2);
    sub_2F8B920(a1, a1 + v6, a2, 0x2E8BA2E8BA2E8BA3LL * (v6 >> 3), 0x2E8BA2E8BA2E8BA3LL * ((a2 - (a1 + v6)) >> 3));
  }
}
