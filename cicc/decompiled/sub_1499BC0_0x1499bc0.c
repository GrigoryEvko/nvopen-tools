// Function: sub_1499BC0
// Address: 0x1499bc0
//
__int64 __fastcall sub_1499BC0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  unsigned __int8 **i; // rbx
  unsigned __int8 *v8; // rsi
  __int64 v9; // rax
  unsigned __int8 **v10; // rdi
  unsigned int v11; // r8d
  _QWORD *v12; // rcx

  result = 8LL * *(unsigned int *)(a1 + 8);
  for ( i = (unsigned __int8 **)(a1 - result); (unsigned __int8 **)a1 != i; ++i )
  {
    v8 = *i;
    result = (unsigned int)**i - 4;
    if ( (unsigned __int8)(**i - 4) <= 0x1Eu )
    {
      v9 = *((unsigned int *)v8 + 2);
      if ( (unsigned int)v9 <= 1 )
      {
        result = 0;
      }
      else
      {
        result = *(_QWORD *)&v8[8 * (1 - v9)];
        if ( result && (unsigned __int8)(*(_BYTE *)result - 4) >= 0x1Fu )
          result = 0;
      }
      if ( a2 == result )
      {
        result = *(_QWORD *)(a3 + 8);
        if ( *(_QWORD *)(a3 + 16) != result )
          goto LABEL_11;
        v10 = (unsigned __int8 **)(result + 8LL * *(unsigned int *)(a3 + 28));
        v11 = *(_DWORD *)(a3 + 28);
        if ( (unsigned __int8 **)result != v10 )
        {
          v12 = 0;
          while ( v8 != *(unsigned __int8 **)result )
          {
            if ( *(_QWORD *)result == -2 )
              v12 = (_QWORD *)result;
            result += 8;
            if ( v10 == (unsigned __int8 **)result )
            {
              if ( !v12 )
                goto LABEL_21;
              *v12 = v8;
              --*(_DWORD *)(a3 + 32);
              ++*(_QWORD *)a3;
              goto LABEL_8;
            }
          }
          continue;
        }
LABEL_21:
        if ( v11 < *(_DWORD *)(a3 + 24) )
        {
          *(_DWORD *)(a3 + 28) = v11 + 1;
          *v10 = v8;
          ++*(_QWORD *)a3;
        }
        else
        {
LABEL_11:
          result = sub_16CCBA0(a3, v8);
        }
      }
    }
LABEL_8:
    ;
  }
  return result;
}
