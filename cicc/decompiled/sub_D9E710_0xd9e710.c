// Function: sub_D9E710
// Address: 0xd9e710
//
__int64 __fastcall sub_D9E710(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r15
  char **v8; // rsi
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r9
  __int64 v14; // r13
  __int64 v15; // r12
  __int64 v16; // r8
  __int64 v17; // r14
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // r12
  __int64 v23; // rdi
  __int64 v24; // rax
  int v25; // r12d
  __int64 v27; // [rsp+8h] [rbp-48h]
  unsigned __int64 v28[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = a1 + 16;
  v8 = (char **)(a1 + 16);
  v10 = sub_C8D7D0(a1, a1 + 16, a2, 0x70u, v28, a6);
  v14 = *(_QWORD *)a1;
  v27 = v10;
  v15 = v10;
  v16 = 112LL * *(unsigned int *)(a1 + 8);
  v17 = *(_QWORD *)a1 + v16;
  if ( *(_QWORD *)a1 != v17 )
  {
    do
    {
      while ( 1 )
      {
        if ( v15 )
        {
          v18 = *(_QWORD *)(v14 + 8);
          *(_QWORD *)(v15 + 16) = 0;
          *(_QWORD *)(v15 + 8) = v18 & 6;
          v19 = *(_QWORD *)(v14 + 24);
          *(_QWORD *)(v15 + 24) = v19;
          LOBYTE(v12) = v19 != -4096;
          LOBYTE(v11) = v19 != 0;
          if ( ((v19 != 0) & (unsigned __int8)v12) != 0 && v19 != -8192 )
          {
            v8 = (char **)(*(_QWORD *)(v14 + 8) & 0xFFFFFFFFFFFFFFF8LL);
            sub_BD6050((unsigned __int64 *)(v15 + 8), (unsigned __int64)v8);
          }
          *(_QWORD *)v15 = &unk_49DE8C0;
          *(_BYTE *)(v15 + 32) = *(_BYTE *)(v14 + 32);
          *(_QWORD *)(v15 + 40) = *(_QWORD *)(v14 + 40);
          *(_QWORD *)(v15 + 48) = *(_QWORD *)(v14 + 48);
          v20 = *(_QWORD *)(v14 + 56);
          *(_DWORD *)(v15 + 72) = 0;
          *(_QWORD *)(v15 + 56) = v20;
          *(_QWORD *)(v15 + 64) = v15 + 80;
          *(_DWORD *)(v15 + 76) = 4;
          if ( *(_DWORD *)(v14 + 72) )
            break;
        }
        v14 += 112;
        v15 += 112;
        if ( v17 == v14 )
          goto LABEL_10;
      }
      v8 = (char **)(v14 + 64);
      v21 = v15 + 64;
      v14 += 112;
      v15 += 112;
      sub_D91460(v21, v8, v11, v12, v16, v13);
    }
    while ( v17 != v14 );
LABEL_10:
    v22 = *(_QWORD *)a1;
    v17 = *(_QWORD *)a1 + 112LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v17 )
    {
      do
      {
        v17 -= 112;
        v23 = *(_QWORD *)(v17 + 64);
        if ( v23 != v17 + 80 )
          _libc_free(v23, v8);
        if ( *(_BYTE *)(v17 + 32) )
          *(_QWORD *)(v17 + 24) = 0;
        v24 = *(_QWORD *)(v17 + 24);
        *(_QWORD *)v17 = &unk_49DB368;
        if ( v24 != -4096 && v24 != 0 && v24 != -8192 )
          sub_BD60C0((_QWORD *)(v17 + 8));
      }
      while ( v17 != v22 );
      v17 = *(_QWORD *)a1;
    }
  }
  v25 = v28[0];
  if ( v7 != v17 )
    _libc_free(v17, v8);
  *(_DWORD *)(a1 + 12) = v25;
  *(_QWORD *)a1 = v27;
  return v27;
}
