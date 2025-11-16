// Function: sub_A730F0
// Address: 0xa730f0
//
char __fastcall sub_A730F0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdi

  v2 = *a1;
  if ( !v2 )
    return a2 != 0;
  if ( a2 )
    return sub_A730E0(v2, a2);
  return 0;
}
