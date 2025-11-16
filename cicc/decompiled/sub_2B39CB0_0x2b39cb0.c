// Function: sub_2B39CB0
// Address: 0x2b39cb0
//
unsigned __int64 __fastcall sub_2B39CB0(__int64 a1, unsigned __int64 a2, int a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 result; // rax
  unsigned __int64 v8; // rdx
  _DWORD *v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // rdx
  unsigned __int64 i; // rdx
  __int64 v13; // rdx

  if ( *(unsigned int *)(a1 + 12) < a2 )
  {
    *(_DWORD *)(a1 + 8) = 0;
    sub_C8D5F0(a1, (const void *)(a1 + 16), a2, 4u, a5, a6);
    result = *(_QWORD *)a1;
    v13 = *(_QWORD *)a1 + 4 * a2;
    if ( *(_QWORD *)a1 != v13 )
    {
      do
      {
        *(_DWORD *)result = a3;
        result += 4LL;
        if ( v13 == result )
          break;
        *(_DWORD *)result = a3;
        result += 4LL;
      }
      while ( v13 != result );
    }
  }
  else
  {
    result = *(unsigned int *)(a1 + 8);
    v8 = a2;
    if ( result <= a2 )
      v8 = *(unsigned int *)(a1 + 8);
    if ( v8 )
    {
      v9 = *(_DWORD **)a1;
      v10 = *(_QWORD *)a1 + 4 * v8;
      do
        *v9++ = a3;
      while ( (_DWORD *)v10 != v9 );
      result = *(unsigned int *)(a1 + 8);
    }
    if ( result < a2 )
    {
      v11 = a2 - result;
      if ( a2 != result )
      {
        result = *(_QWORD *)a1 + 4 * result;
        for ( i = result + 4 * v11; i != result; result += 4LL )
          *(_DWORD *)result = a3;
      }
    }
  }
  *(_DWORD *)(a1 + 8) = a2;
  return result;
}
