// Function: sub_2758850
// Address: 0x2758850
//
__int64 __fastcall sub_2758850(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // r12
  __int64 v10; // rdx
  int v11; // r8d
  __int64 v12; // rsi
  __int64 v13; // rcx
  unsigned __int64 v14; // r13
  unsigned __int64 v15; // r14
  unsigned __int64 v16; // rdi
  int v17; // r13d
  __int64 v19; // [rsp+8h] [rbp-48h]
  unsigned __int64 v20[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v19 = sub_C8D7D0(a1, a1 + 16, a2, 0x38u, v20, a6);
  v8 = *(_QWORD *)a1;
  v9 = *(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 == v9 )
    goto LABEL_13;
  v10 = v19;
  do
  {
    while ( 1 )
    {
      if ( !v10 )
        goto LABEL_4;
      v12 = v10 + 16;
      *(_QWORD *)v10 = *(_QWORD *)v8;
      v13 = *(_QWORD *)(v8 + 24);
      if ( !v13 )
        break;
      v11 = *(_DWORD *)(v8 + 16);
      *(_QWORD *)(v10 + 24) = v13;
      *(_DWORD *)(v10 + 16) = v11;
      *(_QWORD *)(v10 + 32) = *(_QWORD *)(v8 + 32);
      *(_QWORD *)(v10 + 40) = *(_QWORD *)(v8 + 40);
      *(_QWORD *)(v13 + 8) = v12;
      *(_QWORD *)(v10 + 48) = *(_QWORD *)(v8 + 48);
      *(_QWORD *)(v8 + 24) = 0;
      *(_QWORD *)(v8 + 32) = v8 + 16;
      *(_QWORD *)(v8 + 40) = v8 + 16;
      *(_QWORD *)(v8 + 48) = 0;
LABEL_4:
      v8 += 56LL;
      v10 += 56;
      if ( v9 == v8 )
        goto LABEL_8;
    }
    v8 += 56LL;
    *(_DWORD *)(v10 + 16) = 0;
    v10 += 56;
    *(_QWORD *)(v10 - 32) = 0;
    *(_QWORD *)(v10 - 24) = v12;
    *(_QWORD *)(v10 - 16) = v12;
    *(_QWORD *)(v10 - 8) = 0;
  }
  while ( v9 != v8 );
LABEL_8:
  v14 = *(_QWORD *)a1;
  v9 = *(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v9 )
  {
    do
    {
      v15 = *(_QWORD *)(v9 - 32);
      v9 -= 56LL;
      while ( v15 )
      {
        sub_2754510(*(_QWORD *)(v15 + 24));
        v16 = v15;
        v15 = *(_QWORD *)(v15 + 16);
        j_j___libc_free_0(v16);
      }
    }
    while ( v9 != v14 );
    v9 = *(_QWORD *)a1;
  }
LABEL_13:
  v17 = v20[0];
  if ( v6 != v9 )
    _libc_free(v9);
  *(_DWORD *)(a1 + 12) = v17;
  *(_QWORD *)a1 = v19;
  return v19;
}
