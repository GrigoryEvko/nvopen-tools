// Function: sub_16E8750
// Address: 0x16e8750
//
__int64 __fastcall sub_16E8750(__int64 a1, unsigned int a2)
{
  unsigned int v2; // ebx
  unsigned int v3; // r12d

  v2 = a2;
  if ( a2 <= 0x4F )
    return sub_16E7EE0(a1, asc_42AF7C0, a2);
  do
  {
    v3 = 79;
    if ( v2 <= 0x4F )
      v3 = v2;
    sub_16E7EE0(a1, asc_42AF7C0, v3);
    v2 -= v3;
  }
  while ( v2 );
  return a1;
}
