// Function: sub_193DD90
// Address: 0x193dd90
//
__int64 __fastcall sub_193DD90(__int64 a1)
{
  _QWORD *v1; // r12
  _QWORD *v2; // rbx
  unsigned int v3; // r13d
  unsigned int v4; // eax

  v1 = *(_QWORD **)(a1 + 16);
  v2 = *(_QWORD **)(a1 + 8);
  if ( v2 == v1 )
    return 1;
  v3 = 0;
  do
  {
    v4 = sub_193DD90(*v2);
    if ( v3 < v4 )
      v3 = v4;
    ++v2;
  }
  while ( v1 != v2 );
  return v3 + 1;
}
