// Function: sub_31AA620
// Address: 0x31aa620
//
__int64 __fastcall sub_31AA620(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  unsigned __int64 v7; // r12
  __int64 v8; // r9
  unsigned __int64 v9; // r14
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned __int64 v13; // rbx
  __int64 v14; // rax
  int v15; // ebx
  __int64 v17; // [rsp+8h] [rbp-48h]
  unsigned __int64 v18[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = sub_C8D7D0(a1, a1 + 16, a2, 0xB8u, v18, a6);
  v7 = *(_QWORD *)a1;
  v17 = v6;
  v8 = 184LL * *(unsigned int *)(a1 + 8);
  v9 = *(_QWORD *)a1 + v8;
  if ( v7 != v7 + v8 )
  {
    v10 = v6;
    do
    {
      if ( v10 )
      {
        *(_QWORD *)v10 = *(_QWORD *)v7;
        v11 = *(_QWORD *)(v7 + 8);
        *(_QWORD *)(v10 + 16) = 6;
        *(_QWORD *)(v10 + 8) = v11;
        *(_QWORD *)(v10 + 24) = 0;
        v12 = *(_QWORD *)(v7 + 32);
        *(_QWORD *)(v10 + 32) = v12;
        if ( v12 != 0 && v12 != -4096 && v12 != -8192 )
          sub_BD6050((unsigned __int64 *)(v10 + 16), *(_QWORD *)(v7 + 16) & 0xFFFFFFFFFFFFFFF8LL);
        *(_QWORD *)(v10 + 40) = *(_QWORD *)(v7 + 40);
        *(_DWORD *)(v10 + 48) = *(_DWORD *)(v7 + 48);
        *(_DWORD *)(v10 + 52) = *(_DWORD *)(v7 + 52);
        *(_QWORD *)(v10 + 56) = *(_QWORD *)(v7 + 56);
        *(_QWORD *)(v10 + 64) = *(_QWORD *)(v7 + 64);
        *(_BYTE *)(v10 + 72) = *(_BYTE *)(v7 + 72);
        *(_BYTE *)(v10 + 73) = *(_BYTE *)(v7 + 73);
        sub_C8CF70(v10 + 80, (void *)(v10 + 112), 8, v7 + 112, v7 + 80);
        *(_DWORD *)(v10 + 176) = *(_DWORD *)(v7 + 176);
      }
      v7 += 184LL;
      v10 += 184;
    }
    while ( v9 != v7 );
    v13 = *(_QWORD *)a1;
    v9 = *(_QWORD *)a1 + 184LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v9 )
    {
      do
      {
        v9 -= 184LL;
        if ( !*(_BYTE *)(v9 + 108) )
          _libc_free(*(_QWORD *)(v9 + 88));
        v14 = *(_QWORD *)(v9 + 32);
        if ( v14 != 0 && v14 != -4096 && v14 != -8192 )
          sub_BD60C0((_QWORD *)(v9 + 16));
      }
      while ( v9 != v13 );
      v9 = *(_QWORD *)a1;
    }
  }
  v15 = v18[0];
  if ( a1 + 16 != v9 )
    _libc_free(v9);
  *(_DWORD *)(a1 + 12) = v15;
  *(_QWORD *)a1 = v17;
  return v17;
}
