// Function: sub_220EEE0
// Address: 0x220eee0
//
__int64 __fastcall sub_220EEE0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r8
  __int64 v4; // rax
  __int64 v5; // r8

  v1 = *(_QWORD *)(a1 + 24);
  if ( v1 )
  {
    do
    {
      v2 = v1;
      v1 = *(_QWORD *)(v1 + 16);
    }
    while ( v1 );
    return v2;
  }
  v4 = *(_QWORD *)(a1 + 8);
  if ( a1 != *(_QWORD *)(v4 + 24) )
    return *(_QWORD *)(a1 + 8);
  do
  {
    v5 = v4;
    v4 = *(_QWORD *)(v4 + 8);
  }
  while ( *(_QWORD *)(v4 + 24) == v5 );
  if ( v4 != *(_QWORD *)(v5 + 24) )
    return v4;
  return v5;
}
