// Function: sub_1F20BE0
// Address: 0x1f20be0
//
_DWORD *__fastcall sub_1F20BE0(_DWORD *a1, __int64 a2, int *a3, __int64 a4)
{
  __int64 v4; // rsi
  _DWORD *v5; // r8
  __int64 v6; // rax
  int v8; // r11d
  __int64 v9; // rdx
  _DWORD *v10; // rcx

  v4 = a2 - (_QWORD)a1;
  v5 = a1;
  v6 = v4 >> 2;
  if ( v4 <= 0 )
    return a1;
  v8 = *a3;
  do
  {
    while ( 1 )
    {
      v9 = v6 >> 1;
      v10 = &v5[v6 >> 1];
      if ( v8 == -1
        || *v10 != -1
        && *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a4 + 232) + 8LL)
                     + 40LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)(a4 + 232) + 32LL) + v8)
                     + 8) <= *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a4 + 232) + 8LL)
                                       + 40LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)(a4 + 232) + 32LL) + *v10)
                                       + 8) )
      {
        break;
      }
      v6 >>= 1;
      if ( v9 <= 0 )
        return v5;
    }
    v5 = v10 + 1;
    v6 = v6 - v9 - 1;
  }
  while ( v6 > 0 );
  return v5;
}
