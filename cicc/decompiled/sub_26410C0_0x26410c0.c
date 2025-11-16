// Function: sub_26410C0
// Address: 0x26410c0
//
__int64 __fastcall sub_26410C0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  unsigned __int64 v5; // r13
  _QWORD *v6; // r12
  __int64 v7; // r14
  __int64 v8; // rbx
  unsigned __int64 v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rax

  v3 = a2 - (_QWORD)a1;
  v5 = 0x8E38E38E38E38E39LL * (v3 >> 3);
  if ( v3 <= 0 )
    return a3;
  v6 = a1;
  v7 = v3;
  v8 = a3;
  do
  {
    v9 = *(_QWORD *)(v8 + 8);
    *(_QWORD *)v8 = *v6;
    *(_QWORD *)(v8 + 8) = v6[1];
    *(_QWORD *)(v8 + 16) = v6[2];
    *(_QWORD *)(v8 + 24) = v6[3];
    v6[1] = 0;
    v6[2] = 0;
    v6[3] = 0;
    if ( v9 )
      j_j___libc_free_0(v9);
    v10 = *(unsigned int *)(v8 + 64);
    v11 = *(_QWORD *)(v8 + 48);
    *(_QWORD *)(v8 + 32) = v6[4];
    sub_C7D6A0(v11, 4 * v10, 4);
    *(_DWORD *)(v8 + 64) = 0;
    *(_QWORD *)(v8 + 48) = 0;
    *(_DWORD *)(v8 + 56) = 0;
    *(_DWORD *)(v8 + 60) = 0;
    ++*(_QWORD *)(v8 + 40);
    v12 = v6[6];
    v8 += 72;
    ++v6[5];
    v13 = *(_QWORD *)(v8 - 24);
    v6 += 9;
    *(_QWORD *)(v8 - 24) = v12;
    LODWORD(v12) = *((_DWORD *)v6 - 4);
    *(v6 - 3) = v13;
    LODWORD(v13) = *(_DWORD *)(v8 - 16);
    *(_DWORD *)(v8 - 16) = v12;
    LODWORD(v12) = *((_DWORD *)v6 - 3);
    *((_DWORD *)v6 - 4) = v13;
    LODWORD(v13) = *(_DWORD *)(v8 - 12);
    *(_DWORD *)(v8 - 12) = v12;
    LODWORD(v12) = *((_DWORD *)v6 - 2);
    *((_DWORD *)v6 - 3) = v13;
    LODWORD(v13) = *(_DWORD *)(v8 - 8);
    *(_DWORD *)(v8 - 8) = v12;
    *((_DWORD *)v6 - 2) = v13;
    --v5;
  }
  while ( v5 );
  return a3 + v7;
}
