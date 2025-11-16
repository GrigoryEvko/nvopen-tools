// Function: sub_7E6700
// Address: 0x7e6700
//
__int64 __fastcall sub_7E6700(__int64 a1)
{
  __int64 *v1; // rax
  __int64 v2; // r8

  while ( *(_BYTE *)(a1 + 140) == 12 )
    a1 = *(_QWORD *)(a1 + 160);
  v1 = *(__int64 **)(a1 + 168);
  v2 = *v1;
  if ( (v1[2] & 0xC0) == 0x40 )
    return *(_QWORD *)v2;
  return v2;
}
