// Function: sub_C935B0
// Address: 0xc935b0
//
__int64 __fastcall sub_C935B0(_QWORD *a1, unsigned __int8 *a2, __int64 a3, unsigned __int64 a4)
{
  unsigned __int8 *v4; // r8
  __int64 result; // rax
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // r8
  _QWORD v8[4]; // [rsp+0h] [rbp-20h] BYREF

  v4 = &a2[a3];
  result = a4;
  memset(v8, 0, sizeof(v8));
  if ( &a2[a3] != a2 )
  {
    do
    {
      v6 = *a2++;
      v8[v6 >> 6] |= 1LL << v6;
    }
    while ( v4 != a2 );
  }
  v7 = a1[1];
  if ( a4 > v7 )
    result = a1[1];
  if ( a4 >= v7 )
    return -1;
  while ( (v8[(unsigned __int64)*(unsigned __int8 *)(*a1 + result) >> 6] & (1LL << *(_BYTE *)(*a1 + result))) != 0 )
  {
    if ( ++result == v7 )
      return -1;
  }
  return result;
}
