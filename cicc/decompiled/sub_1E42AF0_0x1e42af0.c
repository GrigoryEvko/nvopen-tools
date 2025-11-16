// Function: sub_1E42AF0
// Address: 0x1e42af0
//
__int64 __fastcall sub_1E42AF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  unsigned __int64 v4; // r13
  __int64 v5; // r14
  unsigned __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // r13

  v3 = a2 - a1;
  v4 = 0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 5);
  if ( a2 - a1 <= 0 )
    return a3;
  v5 = a2;
  v6 = 0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 5);
  v7 = a3;
  do
  {
    v8 = *(_QWORD *)(v7 - 88);
    v7 -= 96;
    v5 -= 96;
    j___libc_free_0(v8);
    *(_DWORD *)(v7 + 24) = 0;
    *(_QWORD *)(v7 + 8) = 0;
    *(_DWORD *)(v7 + 16) = 0;
    *(_DWORD *)(v7 + 20) = 0;
    ++*(_QWORD *)v7;
    v9 = *(_QWORD *)(v5 + 8);
    ++*(_QWORD *)v5;
    v10 = *(_QWORD *)(v7 + 8);
    *(_QWORD *)(v7 + 8) = v9;
    LODWORD(v9) = *(_DWORD *)(v5 + 16);
    *(_QWORD *)(v5 + 8) = v10;
    LODWORD(v10) = *(_DWORD *)(v7 + 16);
    *(_DWORD *)(v7 + 16) = v9;
    LODWORD(v9) = *(_DWORD *)(v5 + 20);
    *(_DWORD *)(v5 + 16) = v10;
    LODWORD(v10) = *(_DWORD *)(v7 + 20);
    *(_DWORD *)(v7 + 20) = v9;
    LODWORD(v9) = *(_DWORD *)(v5 + 24);
    *(_DWORD *)(v5 + 20) = v10;
    LODWORD(v10) = *(_DWORD *)(v7 + 24);
    *(_DWORD *)(v7 + 24) = v9;
    *(_DWORD *)(v5 + 24) = v10;
    v11 = *(_QWORD *)(v7 + 32);
    v12 = *(_QWORD *)(v7 + 48);
    *(_QWORD *)(v7 + 32) = *(_QWORD *)(v5 + 32);
    *(_QWORD *)(v7 + 40) = *(_QWORD *)(v5 + 40);
    *(_QWORD *)(v7 + 48) = *(_QWORD *)(v5 + 48);
    *(_QWORD *)(v5 + 32) = 0;
    *(_QWORD *)(v5 + 40) = 0;
    *(_QWORD *)(v5 + 48) = 0;
    if ( v11 )
      j_j___libc_free_0(v11, v12 - v11);
    *(_BYTE *)(v7 + 56) = *(_BYTE *)(v5 + 56);
    *(_DWORD *)(v7 + 60) = *(_DWORD *)(v5 + 60);
    *(_DWORD *)(v7 + 64) = *(_DWORD *)(v5 + 64);
    *(_DWORD *)(v7 + 68) = *(_DWORD *)(v5 + 68);
    *(_DWORD *)(v7 + 72) = *(_DWORD *)(v5 + 72);
    *(_QWORD *)(v7 + 80) = *(_QWORD *)(v5 + 80);
    *(_DWORD *)(v7 + 88) = *(_DWORD *)(v5 + 88);
    --v6;
  }
  while ( v6 );
  v13 = -96;
  v14 = -96LL * v4;
  if ( v3 > 0 )
    v13 = v14;
  return a3 + v13;
}
