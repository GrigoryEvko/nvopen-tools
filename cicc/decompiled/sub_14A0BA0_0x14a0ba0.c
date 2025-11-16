// Function: sub_14A0BA0
// Address: 0x14a0ba0
//
__int64 __fastcall sub_14A0BA0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 result; // rax
  int i; // r14d
  __int64 v8; // r15

  result = a2 + 16;
  if ( a4 )
  {
    for ( i = 0; i != a4; ++i )
    {
      v8 = sub_1643330(a3);
      result = *(unsigned int *)(a2 + 8);
      if ( (unsigned int)result >= *(_DWORD *)(a2 + 12) )
      {
        sub_16CD150(a2, a2 + 16, 0, 8);
        result = *(unsigned int *)(a2 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a2 + 8 * result) = v8;
      ++*(_DWORD *)(a2 + 8);
    }
  }
  return result;
}
