// Function: sub_BD3700
// Address: 0xbd3700
//
__int64 __fastcall sub_BD3700(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r12

  v1 = *(_QWORD *)(a1 + 16);
  if ( !v1 )
    return 0;
  v2 = 0;
  do
  {
    if ( !sub_BD2BE0(*(_QWORD *)(v1 + 24)) )
    {
      if ( v2 )
        return 0;
      v2 = v1;
    }
    v1 = *(_QWORD *)(v1 + 8);
  }
  while ( v1 );
  return v2;
}
