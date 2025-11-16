// Function: sub_FA1E60
// Address: 0xfa1e60
//
__int64 __fastcall sub_FA1E60(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r15
  __int64 v8; // rsi
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rcx
  __int64 v14; // r14
  __int64 result; // rax
  __int64 v16; // r12
  __int64 v17; // rbx
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rbx
  __int64 v23; // rdi
  int v24; // ebx
  __int64 v25; // [rsp+8h] [rbp-48h]
  unsigned __int64 v26[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = a1 + 16;
  v8 = a1 + 16;
  v10 = sub_C8D7D0(a1, a1 + 16, a2, 0x38u, v26, a6);
  v13 = *(unsigned int *)(a1 + 8);
  v14 = v10;
  result = *(_QWORD *)a1;
  v16 = *(_QWORD *)a1 + 56 * v13;
  if ( *(_QWORD *)a1 != v16 )
  {
    v17 = v14;
    do
    {
      while ( 1 )
      {
        if ( v17 )
        {
          v18 = *(_QWORD *)result;
          *(_DWORD *)(v17 + 16) = 0;
          *(_DWORD *)(v17 + 20) = 4;
          *(_QWORD *)v17 = v18;
          *(_QWORD *)(v17 + 8) = v17 + 24;
          v19 = *(unsigned int *)(result + 16);
          if ( (_DWORD)v19 )
            break;
        }
        result += 56;
        v17 += 56;
        if ( v16 == result )
          goto LABEL_7;
      }
      v8 = result + 8;
      v20 = v17 + 8;
      v25 = result;
      v17 += 56;
      sub_F8EFD0(v20, (char **)(result + 8), v19, v13, v11, v12);
      result = v25 + 56;
    }
    while ( v16 != v25 + 56 );
LABEL_7:
    v21 = *(unsigned int *)(a1 + 8);
    v22 = *(_QWORD *)a1;
    result = 7 * v21;
    v16 = *(_QWORD *)a1 + 56 * v21;
    if ( *(_QWORD *)a1 != v16 )
    {
      do
      {
        v16 -= 56;
        v23 = *(_QWORD *)(v16 + 8);
        result = v16 + 24;
        if ( v23 != v16 + 24 )
          result = _libc_free(v23, v8);
      }
      while ( v16 != v22 );
      v16 = *(_QWORD *)a1;
    }
  }
  v24 = v26[0];
  if ( v7 != v16 )
    result = _libc_free(v16, v8);
  *(_QWORD *)a1 = v14;
  *(_DWORD *)(a1 + 12) = v24;
  return result;
}
