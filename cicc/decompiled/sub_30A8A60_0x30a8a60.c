// Function: sub_30A8A60
// Address: 0x30a8a60
//
unsigned __int64 __fastcall sub_30A8A60(_QWORD **a1, __int64 a2)
{
  __int64 v3; // rax
  _QWORD *v4; // rdi
  _QWORD *v5; // rbx
  __int64 v6; // r8
  __int64 v7; // r9
  char v8; // dl
  char v9; // r13
  __int64 v10; // rsi
  unsigned __int64 result; // rax
  const void *v12; // r14
  __int64 v13; // r12
  __int64 v14; // [rsp+0h] [rbp-40h] BYREF
  _BYTE *v15; // [rsp+8h] [rbp-38h]
  __int64 v16; // [rsp+10h] [rbp-30h]
  _BYTE v17[40]; // [rsp+18h] [rbp-28h] BYREF

  v3 = *(_QWORD *)(a2 + 16);
  v4 = *a1;
  v15 = v17;
  v14 = v3;
  v16 = 0x100000000LL;
  v5 = sub_30A8900(v4, &v14);
  v9 = v8;
  if ( v15 != v17 )
    _libc_free((unsigned __int64)v15);
  v10 = *((unsigned int *)v5 + 12);
  if ( v9 )
  {
    v12 = *(const void **)(a2 + 24);
    v13 = *(unsigned int *)(a2 + 32);
    result = *((unsigned int *)v5 + 13);
    if ( v13 + v10 > result )
    {
      result = sub_C8D5F0((__int64)(v5 + 5), v5 + 7, v13 + v10, 8u, v6, v7);
      v10 = *((unsigned int *)v5 + 12);
    }
    if ( 8 * v13 )
    {
      result = (unsigned __int64)memcpy((void *)(v5[5] + 8 * v10), v12, 8 * v13);
      LODWORD(v10) = *((_DWORD *)v5 + 12);
    }
    *((_DWORD *)v5 + 12) = v13 + v10;
  }
  else
  {
    result = 0;
    if ( *((_DWORD *)v5 + 12) )
    {
      do
      {
        *(_QWORD *)(v5[5] + 8 * result) += *(_QWORD *)(*(_QWORD *)(a2 + 24) + 8 * result);
        ++result;
      }
      while ( result != v10 );
    }
  }
  return result;
}
