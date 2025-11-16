// Function: sub_A57470
// Address: 0xa57470
//
_QWORD *__fastcall sub_A57470(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rsi
  __int64 v6; // r12
  _QWORD *result; // rax
  __int64 v8; // rcx
  __int64 v9; // r14
  _QWORD *v10; // rcx
  _QWORD *v11; // rdx
  _QWORD *v12; // r15
  __int64 v13; // rdi
  int v14; // r15d
  _QWORD v15[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = a1 + 16;
  v4 = a1 + 16;
  v6 = sub_C8D7D0(a1, a1 + 16, a2, 32, v15);
  result = *(_QWORD **)a1;
  v8 = 32LL * *(unsigned int *)(a1 + 8);
  v9 = *(_QWORD *)a1 + v8;
  if ( *(_QWORD *)a1 != v9 )
  {
    v10 = (_QWORD *)(v6 + v8);
    v11 = (_QWORD *)v6;
    do
    {
      if ( v11 )
      {
        *v11 = *result;
        v11[1] = result[1];
        v11[2] = result[2];
        v4 = result[3];
        v11[3] = v4;
        result[3] = 0;
        result[2] = 0;
        result[1] = 0;
      }
      v11 += 4;
      result += 4;
    }
    while ( v11 != v10 );
    v12 = *(_QWORD **)a1;
    v9 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v9 )
    {
      do
      {
        v13 = *(_QWORD *)(v9 - 24);
        v9 -= 32;
        if ( v13 )
        {
          v4 = *(_QWORD *)(v9 + 24) - v13;
          result = (_QWORD *)j_j___libc_free_0(v13, v4);
        }
      }
      while ( (_QWORD *)v9 != v12 );
      v9 = *(_QWORD *)a1;
    }
  }
  v14 = v15[0];
  if ( v3 != v9 )
    result = (_QWORD *)_libc_free(v9, v4);
  *(_QWORD *)a1 = v6;
  *(_DWORD *)(a1 + 12) = v14;
  return result;
}
