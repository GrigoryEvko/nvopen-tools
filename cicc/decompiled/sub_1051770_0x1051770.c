// Function: sub_1051770
// Address: 0x1051770
//
__int64 __fastcall sub_1051770(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r14
  __int64 result; // rax
  _QWORD *v13; // r12
  __int64 v14; // rbx
  __int64 v15; // rdx
  _QWORD *v16; // rbx
  int v17; // ebx
  __int64 v18; // [rsp+8h] [rbp-48h]
  unsigned __int64 v19[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = a1 + 16;
  v11 = sub_C8D7D0(a1, a1 + 16, a2, 0x48u, v19, a6);
  result = *(_QWORD *)a1;
  v13 = (_QWORD *)(*(_QWORD *)a1 + 72LL * *(unsigned int *)(a1 + 8));
  if ( *(_QWORD **)a1 != v13 )
  {
    v14 = v11;
    do
    {
      if ( v14 )
      {
        *(_DWORD *)(v14 + 8) = 0;
        *(_QWORD *)v14 = v14 + 16;
        *(_DWORD *)(v14 + 12) = 6;
        v15 = *(unsigned int *)(result + 8);
        if ( (_DWORD)v15 )
        {
          v7 = result;
          v18 = result;
          sub_104D0F0(v14, (char **)result, v15, v8, v9, v10);
          result = v18;
        }
        *(_DWORD *)(v14 + 64) = *(_DWORD *)(result + 64);
      }
      result += 72;
      v14 += 72;
    }
    while ( v13 != (_QWORD *)result );
    v16 = *(_QWORD **)a1;
    result = 9LL * *(unsigned int *)(a1 + 8);
    v13 = (_QWORD *)(*(_QWORD *)a1 + 72LL * *(unsigned int *)(a1 + 8));
    if ( *(_QWORD **)a1 != v13 )
    {
      do
      {
        v13 -= 9;
        result = (__int64)(v13 + 2);
        if ( (_QWORD *)*v13 != v13 + 2 )
          result = _libc_free(*v13, v7);
      }
      while ( v13 != v16 );
      v13 = *(_QWORD **)a1;
    }
  }
  v17 = v19[0];
  if ( (_QWORD *)(a1 + 16) != v13 )
    result = _libc_free(v13, v7);
  *(_QWORD *)a1 = v11;
  *(_DWORD *)(a1 + 12) = v17;
  return result;
}
