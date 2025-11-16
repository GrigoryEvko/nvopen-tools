// Function: sub_2D65B70
// Address: 0x2d65b70
//
unsigned __int64 __fastcall sub_2D65B70(
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
        sub_C8D5F0(a1, (const void *)(a1 + 16), a2, 8u, a5, a6);
        result = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
        for ( i = *(_QWORD *)a1 + 8 * a2; i != result; result += 8LL )
        {
LABEL_5:
          if ( result )
            *(_QWORD *)result = 0;
        }
      }
      else
      {
        result = *(_QWORD *)a1 + 8 * result;
        i = *(_QWORD *)a1 + 8 * a2;
        if ( result != i )
          goto LABEL_5;
      }
    }
    *(_DWORD *)(a1 + 8) = a2;
  }
  return result;
}
