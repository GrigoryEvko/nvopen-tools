// Function: sub_1BB9FC0
// Address: 0x1bb9fc0
//
__int64 __fastcall sub_1BB9FC0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // rax
  _DWORD *v8; // rax
  __int64 i; // rdx
  __int64 result; // rax

  *(_DWORD *)(a3 + 8) = 0;
  if ( a2 )
  {
    v7 = 0;
    if ( a2 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
    {
      sub_16CD150(a3, (const void *)(a3 + 16), a2, 4, a5, a6);
      v7 = 4LL * *(unsigned int *)(a3 + 8);
    }
    v8 = (_DWORD *)(*(_QWORD *)a3 + v7);
    for ( i = *(_QWORD *)a3 + 4LL * a2; (_DWORD *)i != v8; ++v8 )
    {
      if ( v8 )
        *v8 = 0;
    }
    *(_DWORD *)(a3 + 8) = a2;
  }
  result = 0;
  if ( a2 )
  {
    do
    {
      *(_DWORD *)(*(_QWORD *)a3 + 4LL * *(unsigned int *)(a1 + 4 * result)) = result;
      ++result;
    }
    while ( result != a2 );
  }
  return result;
}
