// Function: sub_2B0FC00
// Address: 0x2b0fc00
//
__int64 __fastcall sub_2B0FC00(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdi
  __int64 result; // rax

  *(_DWORD *)(a3 + 8) = 0;
  if ( a2 )
  {
    v8 = 0;
    if ( a2 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
    {
      sub_C8D5F0(a3, (const void *)(a3 + 16), a2, 4u, a5, a6);
      v8 = 4LL * *(unsigned int *)(a3 + 8);
    }
    memset((void *)(*(_QWORD *)a3 + v8), 255, 4LL * a2);
    *(_DWORD *)(a3 + 8) += a2;
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
