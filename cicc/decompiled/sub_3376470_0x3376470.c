// Function: sub_3376470
// Address: 0x3376470
//
unsigned __int64 __fastcall sub_3376470(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
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
        sub_C8D5F0(a1, (const void *)(a1 + 16), a2, 0x10u, a5, a6);
        result = *(unsigned int *)(a1 + 8);
      }
      result = *(_QWORD *)a1 + 16 * result;
      for ( i = *(_QWORD *)a1 + 16 * a2; i != result; result += 16LL )
      {
        if ( result )
        {
          *(_QWORD *)result = 0;
          *(_DWORD *)(result + 8) = 0;
        }
      }
    }
    *(_DWORD *)(a1 + 8) = a2;
  }
  return result;
}
