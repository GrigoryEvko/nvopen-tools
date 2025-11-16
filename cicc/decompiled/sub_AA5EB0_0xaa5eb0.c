// Function: sub_AA5EB0
// Address: 0xaa5eb0
//
__int64 __fastcall sub_AA5EB0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r8

  v1 = sub_AA4FF0(a1);
  if ( !v1 )
    BUG();
  v2 = 0;
  if ( *(_BYTE *)(v1 - 24) == 95 )
    return v1 - 24;
  return v2;
}
