// Function: sub_2B55770
// Address: 0x2b55770
//
__int64 __fastcall sub_2B55770(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  unsigned __int64 v12; // r13
  __int64 v13; // rdi
  unsigned __int64 v14; // r12
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rsi
  unsigned __int64 v21; // r13
  __int64 v22; // rax
  unsigned __int64 v23; // r14
  unsigned __int64 v24; // r15
  unsigned __int64 v25; // rdi
  int v26; // r13d
  __int64 v28; // [rsp+0h] [rbp-50h]
  __int64 v29; // [rsp+8h] [rbp-48h]
  unsigned __int64 v30[7]; // [rsp+18h] [rbp-38h] BYREF

  v29 = a1 + 16;
  v7 = sub_C8D7D0(a1, a1 + 16, a2, 0x38u, v30, a6);
  v11 = *(unsigned int *)(a1 + 8);
  v12 = *(_QWORD *)a1;
  v28 = v7;
  v13 = v7;
  v14 = v12 + 56 * v11;
  if ( v12 != v14 )
  {
    do
    {
      while ( 1 )
      {
        v15 = 56;
        if ( v13 )
        {
          v16 = *(_QWORD *)v12;
          *(_DWORD *)(v13 + 32) = 0;
          v15 = v13 + 56;
          *(_QWORD *)(v13 + 16) = 0;
          *(_DWORD *)(v13 + 24) = 0;
          *(_DWORD *)(v13 + 28) = 0;
          *(_QWORD *)v13 = v16;
          *(_QWORD *)(v13 + 8) = 1;
          v17 = *(_QWORD *)(v12 + 16);
          ++*(_QWORD *)(v12 + 8);
          v18 = *(_QWORD *)(v13 + 16);
          *(_QWORD *)(v13 + 16) = v17;
          LODWORD(v17) = *(_DWORD *)(v12 + 24);
          *(_QWORD *)(v12 + 16) = v18;
          LODWORD(v18) = *(_DWORD *)(v13 + 24);
          *(_DWORD *)(v13 + 24) = v17;
          LODWORD(v17) = *(_DWORD *)(v12 + 28);
          *(_DWORD *)(v12 + 24) = v18;
          LODWORD(v18) = *(_DWORD *)(v13 + 28);
          *(_DWORD *)(v13 + 28) = v17;
          v19 = *(unsigned int *)(v12 + 32);
          *(_DWORD *)(v12 + 28) = v18;
          LODWORD(v18) = *(_DWORD *)(v13 + 32);
          *(_DWORD *)(v13 + 32) = v19;
          *(_DWORD *)(v12 + 32) = v18;
          *(_QWORD *)(v13 + 40) = v13 + 56;
          *(_DWORD *)(v13 + 48) = 0;
          *(_DWORD *)(v13 + 52) = 0;
          if ( *(_DWORD *)(v12 + 48) )
            break;
        }
        v12 += 56LL;
        v13 = v15;
        if ( v14 == v12 )
          goto LABEL_7;
      }
      v20 = v12 + 40;
      v12 += 56LL;
      sub_2B553F0(v13 + 40, v20, v19, v8, v9, v10);
      v13 += 56;
    }
    while ( v14 != v12 );
LABEL_7:
    v21 = *(_QWORD *)a1;
    v14 = *(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v14 )
    {
      do
      {
        v22 = *(unsigned int *)(v14 - 8);
        v23 = *(_QWORD *)(v14 - 16);
        v14 -= 56LL;
        v24 = v23 + 72 * v22;
        if ( v23 != v24 )
        {
          do
          {
            v24 -= 72LL;
            v25 = *(_QWORD *)(v24 + 8);
            if ( v25 != v24 + 24 )
              _libc_free(v25);
          }
          while ( v23 != v24 );
          v23 = *(_QWORD *)(v14 + 40);
        }
        if ( v23 != v14 + 56 )
          _libc_free(v23);
        sub_C7D6A0(*(_QWORD *)(v14 + 16), 16LL * *(unsigned int *)(v14 + 32), 8);
      }
      while ( v14 != v21 );
      v14 = *(_QWORD *)a1;
    }
  }
  v26 = v30[0];
  if ( v29 != v14 )
    _libc_free(v14);
  *(_DWORD *)(a1 + 12) = v26;
  *(_QWORD *)a1 = v28;
  return v28;
}
