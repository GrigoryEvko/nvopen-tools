// Function: sub_9C2620
// Address: 0x9c2620
//
__int64 __fastcall sub_9C2620(__int64 a1, int a2, int a3)
{
  unsigned int v3; // r9d
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rsi
  _QWORD *v11; // rcx

  v3 = *(_DWORD *)(a1 + 8);
  result = v3 - a3;
  LODWORD(v7) = v3 - a3 - a2;
  if ( (unsigned int)result <= (unsigned int)v7 )
  {
    result = (unsigned int)v7;
    v7 = (unsigned int)v7;
    if ( (unsigned int)v7 >= v3 )
      return result;
    goto LABEL_5;
  }
  v8 = (unsigned int)v7;
  v9 = 8LL * (unsigned int)v7;
  v10 = 8 * (v8 + (unsigned int)result + a2 + a3 - v3 - 1 + 1);
  do
  {
    v11 = (_QWORD *)(v9 + *(_QWORD *)a1);
    v9 += 8;
    *v11 |= 2uLL;
  }
  while ( v10 != v9 );
  v7 = (unsigned int)result;
  if ( (unsigned int)result < *(_DWORD *)(a1 + 8) )
  {
    do
    {
LABEL_5:
      *(_QWORD *)(*(_QWORD *)a1 + 8 * v7) |= 4uLL;
      v7 = (unsigned int)(result + 1);
      result = v7;
    }
    while ( (unsigned int)v7 < *(_DWORD *)(a1 + 8) );
  }
  return result;
}
