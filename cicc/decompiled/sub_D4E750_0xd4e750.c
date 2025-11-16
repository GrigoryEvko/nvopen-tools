// Function: sub_D4E750
// Address: 0xd4e750
//
_QWORD *__fastcall sub_D4E750(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  __int64 v8; // rsi
  __int64 v10; // r12
  _QWORD *result; // rax
  __int64 v12; // rdi
  _QWORD *v13; // rdx
  int v14; // r14d
  unsigned __int64 v15[5]; // [rsp+8h] [rbp-28h] BYREF

  v7 = a1 + 16;
  v8 = a1 + 16;
  v10 = sub_C8D7D0(a1, a1 + 16, a2, 0x18u, v15, a6);
  result = *(_QWORD **)a1;
  v12 = *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v12 )
  {
    v13 = (_QWORD *)v10;
    do
    {
      if ( v13 )
      {
        *v13 = *result;
        v13[1] = result[1];
        v13[2] = result[2];
      }
      result += 3;
      v13 += 3;
    }
    while ( (_QWORD *)v12 != result );
    v12 = *(_QWORD *)a1;
  }
  v14 = v15[0];
  if ( v7 != v12 )
    result = (_QWORD *)_libc_free(v12, v8);
  *(_QWORD *)a1 = v10;
  *(_DWORD *)(a1 + 12) = v14;
  return result;
}
