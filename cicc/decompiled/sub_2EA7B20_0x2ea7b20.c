// Function: sub_2EA7B20
// Address: 0x2ea7b20
//
_QWORD *__fastcall sub_2EA7B20(__int64 a1, __int64 a2)
{
  __int64 v3; // r8
  __int64 v4; // r9
  _QWORD *result; // rax
  unsigned int v6; // r13d
  __int64 v7; // rcx
  _QWORD *v8; // rdx
  __int64 v9; // rsi
  _QWORD *i; // rsi

  sub_C8CF70(a1, (void *)(a1 + 32), 8, a2 + 32, a2);
  result = (_QWORD *)(a1 + 112);
  *(_QWORD *)(a1 + 96) = a1 + 112;
  *(_QWORD *)(a1 + 104) = 0x800000000LL;
  v6 = *(_DWORD *)(a2 + 104);
  if ( v6 && a1 + 96 != a2 + 96 )
  {
    v7 = *(_QWORD *)(a2 + 96);
    v8 = (_QWORD *)(a2 + 112);
    if ( v7 == a2 + 112 )
    {
      v9 = v6;
      if ( v6 > 8 )
      {
        sub_2DACD40(a1 + 96, v6, (__int64)v8, v7, v3, v4);
        result = *(_QWORD **)(a1 + 96);
        v8 = *(_QWORD **)(a2 + 96);
        v9 = *(unsigned int *)(a2 + 104);
      }
      for ( i = &v8[3 * v9]; i != v8; result += 3 )
      {
        if ( result )
        {
          *result = *v8;
          result[1] = v8[1];
          result[2] = v8[2];
        }
        v8 += 3;
      }
      *(_DWORD *)(a1 + 104) = v6;
      *(_DWORD *)(a2 + 104) = 0;
    }
    else
    {
      result = (_QWORD *)*(unsigned int *)(a2 + 108);
      *(_DWORD *)(a1 + 104) = v6;
      *(_QWORD *)(a1 + 96) = v7;
      *(_DWORD *)(a1 + 108) = (_DWORD)result;
      *(_QWORD *)(a2 + 96) = v8;
      *(_QWORD *)(a2 + 104) = 0;
    }
  }
  return result;
}
