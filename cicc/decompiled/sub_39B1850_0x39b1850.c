// Function: sub_39B1850
// Address: 0x39b1850
//
__int64 __fastcall sub_39B1850(__int64 a1, __int64 a2, _QWORD *a3, int a4)
{
  __int64 result; // rax
  int i; // r14d
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // r15

  result = a2 + 16;
  if ( a4 )
  {
    for ( i = 0; i != a4; ++i )
    {
      v10 = sub_1643330(a3);
      result = *(unsigned int *)(a2 + 8);
      if ( (unsigned int)result >= *(_DWORD *)(a2 + 12) )
      {
        sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, v8, v9);
        result = *(unsigned int *)(a2 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a2 + 8 * result) = v10;
      ++*(_DWORD *)(a2 + 8);
    }
  }
  return result;
}
