// Function: sub_2640F50
// Address: 0x2640f50
//
__int64 __fastcall sub_2640F50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  unsigned __int64 v4; // r13
  __int64 v5; // r14
  unsigned __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rax
  unsigned __int64 v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // r13

  v3 = a2 - a1;
  v4 = 0x8E38E38E38E38E39LL * ((a2 - a1) >> 3);
  if ( a2 - a1 <= 0 )
    return a3;
  v5 = a2;
  v6 = 0x8E38E38E38E38E39LL * ((a2 - a1) >> 3);
  v7 = a3;
  do
  {
    v8 = *(_QWORD *)(v5 - 72);
    v9 = *(_QWORD *)(v7 - 64);
    v5 -= 72;
    v7 -= 72;
    *(_QWORD *)v7 = v8;
    *(_QWORD *)(v7 + 8) = *(_QWORD *)(v5 + 8);
    *(_QWORD *)(v7 + 16) = *(_QWORD *)(v5 + 16);
    *(_QWORD *)(v7 + 24) = *(_QWORD *)(v5 + 24);
    *(_QWORD *)(v5 + 8) = 0;
    *(_QWORD *)(v5 + 16) = 0;
    *(_QWORD *)(v5 + 24) = 0;
    if ( v9 )
      j_j___libc_free_0(v9);
    v10 = *(unsigned int *)(v7 + 64);
    v11 = *(_QWORD *)(v7 + 48);
    *(_QWORD *)(v7 + 32) = *(_QWORD *)(v5 + 32);
    sub_C7D6A0(v11, 4 * v10, 4);
    *(_DWORD *)(v7 + 64) = 0;
    *(_QWORD *)(v7 + 48) = 0;
    *(_DWORD *)(v7 + 56) = 0;
    *(_DWORD *)(v7 + 60) = 0;
    ++*(_QWORD *)(v7 + 40);
    v12 = *(_QWORD *)(v5 + 48);
    ++*(_QWORD *)(v5 + 40);
    v13 = *(_QWORD *)(v7 + 48);
    *(_QWORD *)(v7 + 48) = v12;
    LODWORD(v12) = *(_DWORD *)(v5 + 56);
    *(_QWORD *)(v5 + 48) = v13;
    LODWORD(v13) = *(_DWORD *)(v7 + 56);
    *(_DWORD *)(v7 + 56) = v12;
    LODWORD(v12) = *(_DWORD *)(v5 + 60);
    *(_DWORD *)(v5 + 56) = v13;
    LODWORD(v13) = *(_DWORD *)(v7 + 60);
    *(_DWORD *)(v7 + 60) = v12;
    LODWORD(v12) = *(_DWORD *)(v5 + 64);
    *(_DWORD *)(v5 + 60) = v13;
    LODWORD(v13) = *(_DWORD *)(v7 + 64);
    *(_DWORD *)(v7 + 64) = v12;
    *(_DWORD *)(v5 + 64) = v13;
    --v6;
  }
  while ( v6 );
  v14 = -72LL * v4;
  if ( v3 <= 0 )
    v14 = -72;
  return v14 + a3;
}
