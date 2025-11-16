// Function: sub_16D25A0
// Address: 0x16d25a0
//
unsigned __int64 __fastcall sub_16D25A0(__int64 *a1, unsigned __int8 *a2, __int64 a3, unsigned __int64 a4)
{
  unsigned __int64 v5; // r10
  unsigned __int8 *v6; // rdi
  unsigned __int64 v7; // rax
  unsigned __int64 result; // rax
  __int64 v9; // r8
  _QWORD v11[4]; // [rsp+0h] [rbp-20h] BYREF

  v5 = a4;
  memset(v11, 0, sizeof(v11));
  if ( a3 )
  {
    v6 = &a2[a3];
    do
    {
      v7 = *a2++;
      v11[v7 >> 6] |= 1LL << v7;
    }
    while ( v6 != a2 );
  }
  if ( a1[1] <= a4 )
    v5 = a1[1];
  result = v5 - 1;
  if ( v5 )
  {
    v9 = *a1;
    do
    {
      if ( (v11[(unsigned __int64)*(unsigned __int8 *)(v9 + result) >> 6] & (1LL << *(_BYTE *)(v9 + result))) != 0 )
        break;
    }
    while ( result-- != 0 );
  }
  return result;
}
