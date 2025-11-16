// Function: sub_32982C0
// Address: 0x32982c0
//
unsigned __int64 __fastcall sub_32982C0(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int64 result; // rax
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rcx
  __int64 v12; // rcx
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // rdx

  result = *(unsigned int *)(a1 + 12);
  if ( result < a2 )
  {
    *(_DWORD *)(a1 + 8) = 0;
    sub_C8D5F0(a1, (const void *)(a1 + 16), a2, 0x10u, a5, a6);
    result = *(_QWORD *)a1;
    v14 = a2;
    do
    {
      if ( result )
      {
        *(_QWORD *)result = a3;
        *(_QWORD *)(result + 8) = a4;
      }
      result += 16LL;
      --v14;
    }
    while ( v14 );
  }
  else
  {
    v10 = *(unsigned int *)(a1 + 8);
    v11 = a2;
    if ( v10 <= a2 )
      v11 = *(unsigned int *)(a1 + 8);
    if ( v11 )
    {
      result = *(_QWORD *)a1;
      v12 = *(_QWORD *)a1 + 16 * v11;
      do
      {
        result += 16LL;
        *(_QWORD *)(result - 16) = a3;
        *(_DWORD *)(result - 8) = a4;
      }
      while ( v12 != result );
      v10 = *(unsigned int *)(a1 + 8);
    }
    if ( a2 > v10 )
    {
      result = *(_QWORD *)a1 + 16 * v10;
      v13 = a2 - v10;
      if ( a2 != v10 )
      {
        do
        {
          if ( result )
          {
            *(_QWORD *)result = a3;
            *(_QWORD *)(result + 8) = a4;
          }
          result += 16LL;
          --v13;
        }
        while ( v13 );
      }
    }
  }
  *(_DWORD *)(a1 + 8) = a2;
  return result;
}
