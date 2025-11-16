// Function: sub_3870860
// Address: 0x3870860
//
__int64 __fastcall sub_3870860(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // rax

  do
  {
    v6 = sub_13FC520(a4);
    v7 = sub_157EBA0(v6);
    v8 = sub_3870570(a1, a3, v7, 0);
    a3 = (__int64 *)v8;
    if ( !v8 )
      return 0;
  }
  while ( v8 != a2 );
  return 1;
}
