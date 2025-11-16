// Function: sub_D97BB0
// Address: 0xd97bb0
//
__int64 __fastcall sub_D97BB0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 v7; // rax

  result = a1[16];
  if ( !result )
    return sub_D970F0(a2);
  v5 = *a1;
  v6 = *a1 + 112LL * *((unsigned int *)a1 + 2);
  if ( v6 != *a1 )
  {
    do
    {
      v7 = *(unsigned int *)(v5 + 72);
      if ( (_DWORD)v7 )
      {
        if ( !a3 )
          return sub_D970F0(a2);
        sub_D91A50(
          a3,
          (char *)(*(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8)),
          *(char **)(v5 + 64),
          (char *)(*(_QWORD *)(v5 + 64) + 8 * v7));
      }
      v5 += 112;
    }
    while ( v6 != v5 );
    return a1[16];
  }
  return result;
}
