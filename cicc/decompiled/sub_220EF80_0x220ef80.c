// Function: sub_220EF80
// Address: 0x220ef80
//
__int64 __fastcall sub_220EF80(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r8
  __int64 v4; // rax

  if ( !*(_DWORD *)a1 && a1 == *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL) )
    return *(_QWORD *)(a1 + 24);
  v1 = *(_QWORD *)(a1 + 16);
  if ( v1 )
  {
    do
    {
      v2 = v1;
      v1 = *(_QWORD *)(v1 + 24);
    }
    while ( v1 );
    return v2;
  }
  v2 = *(_QWORD *)(a1 + 8);
  if ( a1 != *(_QWORD *)(v2 + 16) )
    return v2;
  do
  {
    v4 = v2;
    v2 = *(_QWORD *)(v2 + 8);
  }
  while ( *(_QWORD *)(v2 + 16) == v4 );
  return v2;
}
