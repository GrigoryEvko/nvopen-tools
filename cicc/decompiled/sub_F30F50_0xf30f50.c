// Function: sub_F30F50
// Address: 0xf30f50
//
__int64 __fastcall sub_F30F50(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r15
  __int64 v8; // rsi
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r14
  __int64 result; // rax
  __int64 v15; // r12
  __int64 v16; // rbx
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rbx
  __int64 v20; // rdi
  int v21; // ebx
  __int64 v22; // [rsp+8h] [rbp-48h]
  unsigned __int64 v23[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = a1 + 16;
  v8 = a1 + 16;
  v13 = sub_C8D7D0(a1, a1 + 16, a2, 0x50u, v23, a6);
  result = *(_QWORD *)a1;
  v15 = *(_QWORD *)a1 + 80LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v15 )
  {
    v16 = v13;
    do
    {
      if ( v16 )
      {
        v17 = *(_QWORD *)result;
        *(_DWORD *)(v16 + 16) = 0;
        *(_DWORD *)(v16 + 20) = 6;
        *(_QWORD *)v16 = v17;
        *(_QWORD *)(v16 + 8) = v16 + 24;
        v18 = *(unsigned int *)(result + 16);
        if ( (_DWORD)v18 )
        {
          v8 = result + 8;
          v22 = result;
          sub_F2FA90(v16 + 8, (char **)(result + 8), v18, v10, v11, v12);
          result = v22;
        }
        *(_DWORD *)(v16 + 72) = *(_DWORD *)(result + 72);
      }
      result += 80;
      v16 += 80;
    }
    while ( v15 != result );
    result = *(unsigned int *)(a1 + 8);
    v19 = *(_QWORD *)a1;
    v15 = *(_QWORD *)a1 + 80 * result;
    if ( *(_QWORD *)a1 != v15 )
    {
      do
      {
        v15 -= 80;
        v20 = *(_QWORD *)(v15 + 8);
        result = v15 + 24;
        if ( v20 != v15 + 24 )
          result = _libc_free(v20, v8);
      }
      while ( v15 != v19 );
      v15 = *(_QWORD *)a1;
    }
  }
  v21 = v23[0];
  if ( v7 != v15 )
    result = _libc_free(v15, v8);
  *(_QWORD *)a1 = v13;
  *(_DWORD *)(a1 + 12) = v21;
  return result;
}
