// Function: sub_BD3750
// Address: 0xbd3750
//
__int64 __fastcall sub_BD3750(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r13
  __int64 v3; // r12

  v1 = *(_QWORD *)(a1 + 16);
  if ( !v1 )
    return 0;
  v2 = 0;
  do
  {
    v3 = *(_QWORD *)(v1 + 24);
    if ( !sub_BD2BE0(v3) )
    {
      if ( v2 && v3 != v2 )
        return 0;
      v2 = v3;
    }
    v1 = *(_QWORD *)(v1 + 8);
  }
  while ( v1 );
  return v2;
}
