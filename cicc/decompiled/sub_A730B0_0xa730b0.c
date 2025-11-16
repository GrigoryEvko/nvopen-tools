// Function: sub_A730B0
// Address: 0xa730b0
//
__int64 __fastcall sub_A730B0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdi

  v2 = *a1;
  if ( !v2 )
    return a2 != 0;
  if ( a2 )
    return sub_A72F30(v2, a2, 1);
  return 0xFFFFFFFFLL;
}
