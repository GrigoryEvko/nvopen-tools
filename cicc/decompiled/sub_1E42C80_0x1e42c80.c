// Function: sub_1E42C80
// Address: 0x1e42c80
//
__int64 __fastcall sub_1E42C80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  unsigned __int64 v5; // r13
  __int64 v6; // r12
  __int64 v7; // r14
  __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rsi
  char v13; // al

  v3 = a2 - a1;
  v5 = 0xAAAAAAAAAAAAAAABLL * (v3 >> 5);
  if ( v3 <= 0 )
    return a3;
  v6 = a1;
  v7 = v3;
  v8 = a3;
  do
  {
    j___libc_free_0(*(_QWORD *)(v8 + 8));
    *(_DWORD *)(v8 + 24) = 0;
    *(_QWORD *)(v8 + 8) = 0;
    *(_DWORD *)(v8 + 16) = 0;
    *(_DWORD *)(v8 + 20) = 0;
    ++*(_QWORD *)v8;
    v9 = *(_QWORD *)(v6 + 8);
    ++*(_QWORD *)v6;
    v10 = *(_QWORD *)(v8 + 8);
    *(_QWORD *)(v8 + 8) = v9;
    LODWORD(v9) = *(_DWORD *)(v6 + 16);
    *(_QWORD *)(v6 + 8) = v10;
    LODWORD(v10) = *(_DWORD *)(v8 + 16);
    *(_DWORD *)(v8 + 16) = v9;
    LODWORD(v9) = *(_DWORD *)(v6 + 20);
    *(_DWORD *)(v6 + 16) = v10;
    LODWORD(v10) = *(_DWORD *)(v8 + 20);
    *(_DWORD *)(v8 + 20) = v9;
    LODWORD(v9) = *(_DWORD *)(v6 + 24);
    *(_DWORD *)(v6 + 20) = v10;
    LODWORD(v10) = *(_DWORD *)(v8 + 24);
    *(_DWORD *)(v8 + 24) = v9;
    *(_DWORD *)(v6 + 24) = v10;
    v11 = *(_QWORD *)(v8 + 32);
    v12 = *(_QWORD *)(v8 + 48);
    *(_QWORD *)(v8 + 32) = *(_QWORD *)(v6 + 32);
    *(_QWORD *)(v8 + 40) = *(_QWORD *)(v6 + 40);
    *(_QWORD *)(v8 + 48) = *(_QWORD *)(v6 + 48);
    *(_QWORD *)(v6 + 32) = 0;
    *(_QWORD *)(v6 + 40) = 0;
    *(_QWORD *)(v6 + 48) = 0;
    if ( v11 )
      j_j___libc_free_0(v11, v12 - v11);
    v13 = *(_BYTE *)(v6 + 56);
    v8 += 96;
    v6 += 96;
    *(_BYTE *)(v8 - 40) = v13;
    *(_DWORD *)(v8 - 36) = *(_DWORD *)(v6 - 36);
    *(_DWORD *)(v8 - 32) = *(_DWORD *)(v6 - 32);
    *(_DWORD *)(v8 - 28) = *(_DWORD *)(v6 - 28);
    *(_DWORD *)(v8 - 24) = *(_DWORD *)(v6 - 24);
    *(_QWORD *)(v8 - 16) = *(_QWORD *)(v6 - 16);
    *(_DWORD *)(v8 - 8) = *(_DWORD *)(v6 - 8);
    --v5;
  }
  while ( v5 );
  return a3 + v7;
}
