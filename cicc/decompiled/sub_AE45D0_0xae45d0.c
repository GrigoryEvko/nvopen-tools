// Function: sub_AE45D0
// Address: 0xae45d0
//
__int64 __fastcall sub_AE45D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // r13

  v2 = *(unsigned int *)(a1 + 8);
  if ( v2 != *(_DWORD *)(a2 + 8) )
    return 0;
  v4 = *(_QWORD *)a1;
  v5 = *(_QWORD *)a2;
  v6 = *(_QWORD *)a1 + 8 * v2;
  if ( *(_QWORD *)a1 == v6 )
    return 1;
  while ( sub_AE1CE0(v4, v5) )
  {
    v4 += 8;
    v5 += 8;
    if ( v6 == v4 )
      return 1;
  }
  return 0;
}
