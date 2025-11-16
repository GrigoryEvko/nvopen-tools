// Function: sub_B91360
// Address: 0xb91360
//
void __fastcall sub_B91360(__int64 *a1)
{
  __int64 *v1; // rbx

  v1 = a1;
  do
  {
    if ( *v1 )
      sub_B91220((__int64)v1, *v1);
    ++v1;
  }
  while ( v1 != a1 + 3 );
}
