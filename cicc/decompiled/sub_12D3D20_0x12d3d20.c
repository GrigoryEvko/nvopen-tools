// Function: sub_12D3D20
// Address: 0x12d3d20
//
__int64 __fastcall sub_12D3D20(__int64 a1)
{
  int v1; // r14d
  __int64 v2; // r13
  int v3; // r12d
  __int64 v4; // rbx
  __int64 v5; // rdi
  __int64 v6; // rdi

  v1 = 0;
  v2 = a1 + 72;
  v3 = 0;
  v4 = *(_QWORD *)(a1 + 80);
  if ( v4 == a1 + 72 )
    return 0;
  do
  {
    v5 = v4 - 24;
    if ( !v4 )
      v5 = 0;
    ++v3;
    v6 = sub_157EBA0(v5);
    if ( v6 )
      v1 += sub_15F4D60(v6);
    v4 = *(_QWORD *)(v4 + 8);
  }
  while ( v2 != v4 );
  if ( !v3 )
    return 0;
  else
    return (unsigned int)(v1 + 2 - v3);
}
