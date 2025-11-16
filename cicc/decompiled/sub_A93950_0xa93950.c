// Function: sub_A93950
// Address: 0xa93950
//
unsigned __int64 __fastcall sub_A93950(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 result; // rax
  __int64 i; // rdx

  result = *(unsigned int *)(a1 + 8);
  if ( a2 != result )
  {
    if ( a2 >= result )
    {
      if ( a2 > *(unsigned int *)(a1 + 12) )
      {
        sub_C8D5F0(a1, a1 + 16, a2, 4);
        result = *(_QWORD *)a1 + 4LL * *(unsigned int *)(a1 + 8);
        for ( i = *(_QWORD *)a1 + 4 * a2; i != result; result += 4LL )
        {
LABEL_5:
          if ( result )
            *(_DWORD *)result = 0;
        }
      }
      else
      {
        result = *(_QWORD *)a1 + 4 * result;
        i = *(_QWORD *)a1 + 4 * a2;
        if ( result != i )
          goto LABEL_5;
      }
    }
    *(_DWORD *)(a1 + 8) = a2;
  }
  return result;
}
