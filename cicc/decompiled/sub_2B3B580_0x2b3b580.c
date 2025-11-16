// Function: sub_2B3B580
// Address: 0x2b3b580
//
unsigned __int64 __fastcall sub_2B3B580(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rsi
  unsigned __int64 result; // rax
  unsigned __int64 v10; // rdx

  v8 = *(unsigned int *)(a1 + 8);
  result = *(unsigned int *)(a1 + 12);
  if ( a2 + v8 > result )
  {
    result = sub_C8D5F0(a1, (const void *)(a1 + 16), a2 + v8, 8u, a5, a6);
    v8 = *(unsigned int *)(a1 + 8);
  }
  if ( a2 )
  {
    result = *(_QWORD *)a1 + 8 * v8;
    v10 = result + 8 * a2;
    if ( result != v10 )
    {
      do
      {
        *(_QWORD *)result = a3;
        result += 8LL;
      }
      while ( v10 != result );
      LODWORD(v8) = *(_DWORD *)(a1 + 8);
    }
  }
  *(_DWORD *)(a1 + 8) = v8 + a2;
  return result;
}
