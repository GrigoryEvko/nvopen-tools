// Function: sub_1DF38A0
// Address: 0x1df38a0
//
unsigned int *__fastcall sub_1DF38A0(__int64 a1, __int64 a2)
{
  unsigned int *result; // rax
  unsigned int *v5; // rdx
  unsigned int v6; // esi
  _DWORD *v7; // rdi
  int v8; // ecx

  result = (unsigned int *)*(unsigned int *)(a1 + 16);
  if ( (_DWORD)result )
  {
    result = *(unsigned int **)(a1 + 8);
    v5 = &result[4 * *(unsigned int *)(a1 + 24)];
    if ( result != v5 )
    {
      while ( 1 )
      {
        v6 = *result;
        v7 = result;
        if ( *result <= 0xFFFFFFFD )
          break;
        result += 4;
        if ( v5 == result )
          return result;
      }
      if ( result != v5 )
      {
        while ( 1 )
        {
          for ( result = v7 + 4; v5 != result; result += 4 )
          {
            if ( *result <= 0xFFFFFFFD )
              break;
          }
          v8 = *(_DWORD *)(*(_QWORD *)(a2 + 24) + 4LL * (v6 >> 5));
          if ( !_bittest(&v8, v6) )
          {
            *v7 = -2;
            --*(_DWORD *)(a1 + 16);
            ++*(_DWORD *)(a1 + 20);
          }
          if ( result == v5 )
            break;
          v6 = *result;
          v7 = result;
        }
      }
    }
  }
  return result;
}
