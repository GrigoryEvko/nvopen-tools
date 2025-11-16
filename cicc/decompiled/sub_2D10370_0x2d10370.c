// Function: sub_2D10370
// Address: 0x2d10370
//
__int64 __fastcall sub_2D10370(__int64 a1, int a2, unsigned __int8 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  unsigned int v9; // r12d
  __int64 *v10; // rsi
  __int64 v11; // r13
  unsigned int v12; // r12d
  int v13; // r14d
  __int64 v14; // rdx

  result = 0x600000000LL;
  v9 = a2 + 63;
  v10 = (__int64 *)(a1 + 16);
  v11 = -(__int64)a3;
  v12 = v9 >> 6;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x600000000LL;
  if ( v12 > 6 )
  {
    sub_C8D5F0(a1, v10, v12, 8u, a5, a6);
    result = *(_QWORD *)a1;
    v14 = *(_QWORD *)a1 + 8LL * v12;
    do
    {
      *(_QWORD *)result = v11;
      result += 8;
    }
    while ( v14 != result );
  }
  else if ( v12 )
  {
    result = (__int64)&v10[v12];
    do
      *v10++ = v11;
    while ( (__int64 *)result != v10 );
  }
  *(_DWORD *)(a1 + 8) = v12;
  *(_DWORD *)(a1 + 64) = a2;
  if ( a3 )
  {
    v13 = a2 & 0x3F;
    if ( v13 )
    {
      result = ~(-1LL << v13);
      *(_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - 8) &= result;
    }
  }
  return result;
}
