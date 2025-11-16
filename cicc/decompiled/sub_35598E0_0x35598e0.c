// Function: sub_35598E0
// Address: 0x35598e0
//
void __fastcall sub_35598E0(char *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r9

  if ( a2 - (__int64)a1 <= 1232 )
  {
    sub_35407B0((__int64)a1, a2);
  }
  else
  {
    v2 = 88 * ((0x2E8BA2E8BA2E8BA3LL * ((a2 - (__int64)a1) >> 3)) >> 1);
    sub_35598E0(a1, &a1[v2]);
    sub_35598E0(&a1[v2], a2);
    sub_3559700(
      a1,
      &a1[v2],
      a2,
      0x2E8BA2E8BA2E8BA3LL * (v2 >> 3),
      0x2E8BA2E8BA2E8BA3LL * ((a2 - (__int64)&a1[v2]) >> 3),
      v3);
  }
}
