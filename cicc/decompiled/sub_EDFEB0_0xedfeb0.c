// Function: sub_EDFEB0
// Address: 0xedfeb0
//
_QWORD *__fastcall sub_EDFEB0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  __int64 v8; // rsi
  __int64 v10; // r12
  _QWORD *result; // rax
  __int64 v12; // rcx
  __int64 v13; // r14
  _QWORD *v14; // rcx
  _QWORD *v15; // rdx
  _QWORD *v16; // r15
  __int64 v17; // rdi
  int v18; // r15d
  unsigned __int64 v19[7]; // [rsp+8h] [rbp-38h] BYREF

  v7 = a1 + 16;
  v8 = a1 + 16;
  v10 = sub_C8D7D0(a1, a1 + 16, a2, 0x20u, v19, a6);
  result = *(_QWORD **)a1;
  v12 = 32LL * *(unsigned int *)(a1 + 8);
  v13 = *(_QWORD *)a1 + v12;
  if ( *(_QWORD *)a1 != v13 )
  {
    v14 = (_QWORD *)(v10 + v12);
    v15 = (_QWORD *)v10;
    do
    {
      if ( v15 )
      {
        *v15 = *result;
        v15[1] = result[1];
        v15[2] = result[2];
        v8 = result[3];
        result[2] = 0;
        result[1] = 0;
        *result = 0;
        v15[3] = v8;
      }
      v15 += 4;
      result += 4;
    }
    while ( v15 != v14 );
    v16 = *(_QWORD **)a1;
    v13 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v13 )
    {
      do
      {
        v17 = *(_QWORD *)(v13 - 32);
        v13 -= 32;
        if ( v17 )
        {
          v8 = *(_QWORD *)(v13 + 16) - v17;
          result = (_QWORD *)j_j___libc_free_0(v17, v8);
        }
      }
      while ( (_QWORD *)v13 != v16 );
      v13 = *(_QWORD *)a1;
    }
  }
  v18 = v19[0];
  if ( v7 != v13 )
    result = (_QWORD *)_libc_free(v13, v8);
  *(_QWORD *)a1 = v10;
  *(_DWORD *)(a1 + 12) = v18;
  return result;
}
