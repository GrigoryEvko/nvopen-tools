// Function: sub_25535D0
// Address: 0x25535d0
//
__int64 __fastcall sub_25535D0(__int64 a1, __int64 *a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  int v8; // r13d
  const void *v10; // rsi
  __int64 v11; // r14
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rcx
  __int64 v14; // r13
  int v15; // edx

  result = 0x300000000LL;
  v8 = a3;
  v10 = (const void *)(a1 + 16);
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x300000000LL;
  if ( a3 > 3 )
  {
    result = sub_C8D5F0(a1, v10, a3, 0x10u, a5, a6);
    v10 = (const void *)(a1 + 16);
  }
  if ( v8 )
  {
    v11 = (__int64)&a2[(unsigned int)(v8 - 1) + 1];
    do
    {
      v12 = *(unsigned int *)(a1 + 8);
      v13 = *(unsigned int *)(a1 + 12);
      v14 = *a2;
      v15 = *(_DWORD *)(a1 + 8);
      if ( v12 >= v13 )
      {
        if ( v13 < v12 + 1 )
        {
          sub_C8D5F0(a1, v10, v12 + 1, 0x10u, a5, a6);
          v12 = *(unsigned int *)(a1 + 8);
        }
        result = *(_QWORD *)a1 + 16 * v12;
        *(_QWORD *)result = v14;
        *(_QWORD *)(result + 8) = a4;
        ++*(_DWORD *)(a1 + 8);
      }
      else
      {
        result = *(_QWORD *)a1 + 16 * v12;
        if ( result )
        {
          *(_QWORD *)result = v14;
          *(_QWORD *)(result + 8) = a4;
          v15 = *(_DWORD *)(a1 + 8);
        }
        *(_DWORD *)(a1 + 8) = v15 + 1;
      }
      ++a2;
    }
    while ( (__int64 *)v11 != a2 );
  }
  return result;
}
