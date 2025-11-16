// Function: sub_D97B30
// Address: 0xd97b30
//
__int64 __fastcall sub_D97B30(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r12
  __int64 v8; // rcx

  v5 = *a1;
  v6 = *a1 + 112LL * *((unsigned int *)a1 + 2);
  if ( v6 == *a1 )
    return 0;
  while ( 1 )
  {
    v7 = v5;
    if ( a2 == *(_QWORD *)(v5 + 24) )
    {
      v8 = *(unsigned int *)(v5 + 72);
      if ( !(_DWORD)v8 )
        return v7;
      if ( a3 )
        break;
    }
    v5 += 112;
    if ( v6 == v5 )
      return 0;
  }
  sub_D91A50(
    a3,
    (char *)(*(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8)),
    *(char **)(v5 + 64),
    (char *)(*(_QWORD *)(v5 + 64) + 8 * v8));
  return v7;
}
