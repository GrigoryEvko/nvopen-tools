// Function: sub_D354F0
// Address: 0xd354f0
//
__int64 __fastcall sub_D354F0(__int64 a1)
{
  unsigned int v1; // eax

  v1 = *(_DWORD *)(a1 + 8);
  if ( v1 <= 4 )
    return 0;
  if ( v1 - 5 > 2 )
    BUG();
  return 1;
}
