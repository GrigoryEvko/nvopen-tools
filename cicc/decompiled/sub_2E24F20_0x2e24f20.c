// Function: sub_2E24F20
// Address: 0x2e24f20
//
__int64 __fastcall sub_2E24F20(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx

  v2 = *(_QWORD *)(a1 + 32);
  v3 = *(_QWORD *)(a1 + 40);
  if ( v2 == v3 )
    return 0;
  while ( a2 != *(_QWORD *)(*(_QWORD *)v2 + 24LL) )
  {
    v2 += 8;
    if ( v3 == v2 )
      return 0;
  }
  return *(_QWORD *)v2;
}
