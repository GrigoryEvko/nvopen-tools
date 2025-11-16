// Function: sub_31AA470
// Address: 0x31aa470
//
__int64 __fastcall sub_31AA470(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // r15
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rbx
  __int64 v13; // rcx
  _QWORD *v14; // r12
  _QWORD *v15; // r14
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rdx
  char **v20; // rsi
  __int64 v21; // rdi
  _QWORD *v22; // rbx
  unsigned __int64 v23; // rdi
  __int64 v24; // rax
  int v25; // ebx
  __int64 v27; // [rsp+8h] [rbp-48h]
  unsigned __int64 v28[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = (_QWORD *)(a1 + 16);
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x58u, v28, a6);
  v11 = *(unsigned int *)(a1 + 8);
  v27 = v8;
  v12 = v8;
  v13 = 5 * v11;
  v14 = (_QWORD *)(*(_QWORD *)a1 + 88 * v11);
  if ( *(_QWORD **)a1 != v14 )
  {
    v15 = *(_QWORD **)a1;
    do
    {
      while ( 1 )
      {
        if ( v12 )
        {
          v16 = *v15;
          *(_QWORD *)(v12 + 8) = 6;
          *(_QWORD *)(v12 + 16) = 0;
          *(_QWORD *)v12 = v16;
          v17 = v15[3];
          *(_QWORD *)(v12 + 24) = v17;
          LOBYTE(v13) = v17 != 0;
          if ( v17 != 0 && v17 != -4096 && v17 != -8192 )
            sub_BD6050((unsigned __int64 *)(v12 + 8), v15[1] & 0xFFFFFFFFFFFFFFF8LL);
          *(_DWORD *)(v12 + 32) = *((_DWORD *)v15 + 8);
          *(_QWORD *)(v12 + 40) = v15[5];
          v18 = v15[6];
          *(_DWORD *)(v12 + 64) = 0;
          *(_QWORD *)(v12 + 48) = v18;
          v19 = v12 + 72;
          *(_QWORD *)(v12 + 56) = v12 + 72;
          *(_DWORD *)(v12 + 68) = 2;
          if ( *((_DWORD *)v15 + 16) )
            break;
        }
        v15 += 11;
        v12 += 88;
        if ( v14 == v15 )
          goto LABEL_10;
      }
      v20 = (char **)(v15 + 7);
      v21 = v12 + 56;
      v15 += 11;
      v12 += 88;
      sub_31A3A30(v21, v20, v19, v13, v9, v10);
    }
    while ( v14 != v15 );
LABEL_10:
    v22 = *(_QWORD **)a1;
    v14 = (_QWORD *)(*(_QWORD *)a1 + 88LL * *(unsigned int *)(a1 + 8));
    if ( *(_QWORD **)a1 != v14 )
    {
      do
      {
        v14 -= 11;
        v23 = v14[7];
        if ( (_QWORD *)v23 != v14 + 9 )
          _libc_free(v23);
        v24 = v14[3];
        if ( v24 != 0 && v24 != -4096 && v24 != -8192 )
          sub_BD60C0(v14 + 1);
      }
      while ( v14 != v22 );
      v14 = *(_QWORD **)a1;
    }
  }
  v25 = v28[0];
  if ( v6 != v14 )
    _libc_free((unsigned __int64)v14);
  *(_DWORD *)(a1 + 12) = v25;
  *(_QWORD *)a1 = v27;
  return v27;
}
