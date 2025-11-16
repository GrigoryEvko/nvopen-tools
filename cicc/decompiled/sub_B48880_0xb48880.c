// Function: sub_B48880
// Address: 0xb48880
//
__int64 __fastcall sub_B48880(__int64 *a1, unsigned int a2, unsigned __int8 a3)
{
  __int64 result; // rax
  __int64 v6; // r14
  _QWORD *v7; // rsi
  unsigned int v8; // r15d
  int v9; // ebx
  __int64 v10; // rdx

  *a1 = 1;
  if ( a2 > 0x39 )
  {
    result = sub_22077B0(72);
    v6 = result;
    if ( result )
    {
      v7 = (_QWORD *)(result + 16);
      *(_QWORD *)result = result + 16;
      v8 = (a2 + 63) >> 6;
      result = 0x600000000LL;
      *(_QWORD *)(v6 + 8) = 0x600000000LL;
      if ( v8 > 6 )
      {
        sub_C8D5F0(v6, v7, v8, 8);
        result = *(_QWORD *)v6;
        v10 = *(_QWORD *)v6 + 8LL * v8;
        do
        {
          *(_QWORD *)result = -(__int64)a3;
          result += 8;
        }
        while ( v10 != result );
      }
      else if ( v8 )
      {
        result = (__int64)&v7[v8];
        do
          *v7++ = -(__int64)a3;
        while ( (_QWORD *)result != v7 );
      }
      *(_DWORD *)(v6 + 8) = v8;
      *(_DWORD *)(v6 + 64) = a2;
      if ( a3 )
      {
        v9 = a2 & 0x3F;
        if ( v9 )
        {
          result = ~(-1LL << v9);
          *(_QWORD *)(*(_QWORD *)v6 + 8LL * *(unsigned int *)(v6 + 8) - 8) &= result;
        }
      }
    }
    *a1 = v6;
  }
  else
  {
    result = 2 * (((unsigned __int64)a2 << 57) | -(__int64)a3 & ~(-1LL << a2)) + 1;
    *a1 = result;
  }
  return result;
}
