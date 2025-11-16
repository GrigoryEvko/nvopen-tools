// Function: sub_E9F5A0
// Address: 0xe9f5a0
//
__int64 __fastcall sub_E9F5A0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v5; // r9
  __int64 v7; // r8
  __int64 v8; // rbx
  __int64 result; // rax
  __int64 v10; // [rsp+0h] [rbp-40h]
  __int64 v11; // [rsp+8h] [rbp-38h]

  v4 = a3 + (a4 << 6);
  if ( a3 != v4 )
  {
    v5 = a4;
    v7 = 1LL << a2;
    v8 = a3;
    do
    {
      result = *(_QWORD *)(v8 + 8LL * (a2 >> 6) + 24) & v7;
      if ( result )
      {
        v10 = v7;
        v11 = v5;
        *(_QWORD *)(a1 + 8LL * (*(_DWORD *)(v8 + 16) >> 6)) &= ~(1LL << *(_DWORD *)(v8 + 16));
        result = sub_E9F5A0(a1, *(unsigned int *)(v8 + 16), a3, v5);
        v7 = v10;
        v5 = v11;
      }
      v8 += 64;
    }
    while ( v4 != v8 );
  }
  return result;
}
