// Function: sub_1D61E90
// Address: 0x1d61e90
//
unsigned __int64 __fastcall sub_1D61E90(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned __int64 result; // rax
  __int64 i; // rdx

  result = *(unsigned int *)(a1 + 8);
  if ( a2 >= result )
  {
    if ( a2 <= result )
      return result;
    if ( a2 > *(unsigned int *)(a1 + 12) )
    {
      sub_16CD150(a1, (const void *)(a1 + 16), a2, 8, a5, a6);
      result = *(unsigned int *)(a1 + 8);
    }
    result = *(_QWORD *)a1 + 8 * result;
    for ( i = *(_QWORD *)a1 + 8 * a2; i != result; result += 8LL )
    {
      if ( result )
        *(_QWORD *)result = 0;
    }
  }
  *(_DWORD *)(a1 + 8) = a2;
  return result;
}
