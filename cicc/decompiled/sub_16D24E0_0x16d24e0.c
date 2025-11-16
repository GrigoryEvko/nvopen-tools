// Function: sub_16D24E0
// Address: 0x16d24e0
//
__int64 __fastcall sub_16D24E0(_QWORD *a1, unsigned __int8 *a2, __int64 a3, unsigned __int64 a4)
{
  unsigned __int8 *v5; // rdi
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdi
  __int64 result; // rax
  _QWORD v9[4]; // [rsp+0h] [rbp-20h] BYREF

  memset(v9, 0, sizeof(v9));
  if ( a3 )
  {
    v5 = &a2[a3];
    do
    {
      v6 = *a2++;
      v9[v6 >> 6] |= 1LL << v6;
    }
    while ( v5 != a2 );
  }
  v7 = a1[1];
  result = -1;
  if ( a4 < v7 )
  {
    result = a4;
    while ( (v9[(unsigned __int64)*(unsigned __int8 *)(*a1 + result) >> 6] & (1LL << *(_BYTE *)(*a1 + result))) != 0 )
    {
      if ( ++result == v7 )
        return -1;
    }
  }
  return result;
}
