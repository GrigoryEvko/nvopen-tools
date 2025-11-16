// Function: sub_3157150
// Address: 0x3157150
//
__int64 __fastcall sub_3157150(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // ebx
  unsigned __int64 v7; // rdx
  __int64 result; // rax

  if ( *(_DWORD *)(a1 + 12) < a2 )
    result = sub_C8D5F0(a1, (const void *)(a1 + 16), a2, 4u, a5, a6);
  v6 = *(_DWORD *)(a1 + 8);
  if ( a2 > v6 )
  {
    do
    {
      v7 = v6 + 1LL;
      if ( v7 > *(unsigned int *)(a1 + 12) )
        sub_C8D5F0(a1, (const void *)(a1 + 16), v7, 4u, a5, a6);
      *(_DWORD *)(*(_QWORD *)a1 + 4LL * *(unsigned int *)(a1 + 8)) = v6;
      result = *(unsigned int *)(a1 + 8);
      v6 = result + 1;
      *(_DWORD *)(a1 + 8) = result + 1;
    }
    while ( a2 > (int)result + 1 );
  }
  return result;
}
