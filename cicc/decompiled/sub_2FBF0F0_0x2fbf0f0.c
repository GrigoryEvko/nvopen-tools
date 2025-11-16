// Function: sub_2FBF0F0
// Address: 0x2fbf0f0
//
_DWORD *__fastcall sub_2FBF0F0(_DWORD *a1, __int64 a2, int *a3, __int64 a4)
{
  __int64 v4; // rsi
  _DWORD *v5; // r8
  __int64 v6; // rax
  int v7; // r10d
  __int64 v8; // rdx
  _DWORD *v9; // rsi

  v4 = a2 - (_QWORD)a1;
  v5 = a1;
  v6 = v4 >> 2;
  if ( v4 <= 0 )
    return a1;
  v7 = *a3;
  do
  {
    while ( 1 )
    {
      v8 = v6 >> 1;
      v9 = &v5[v6 >> 1];
      if ( *v9 == -1
        || v7 != -1
        && *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a4 + 8LL)
                     + 40LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)a4 + 32LL) + *v9)
                     + 8) <= *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a4 + 8LL)
                                       + 40LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)a4 + 32LL) + v7)
                                       + 8) )
      {
        break;
      }
      v5 = v9 + 1;
      v6 = v6 - v8 - 1;
      if ( v6 <= 0 )
        return v5;
    }
    v6 >>= 1;
  }
  while ( v8 > 0 );
  return v5;
}
