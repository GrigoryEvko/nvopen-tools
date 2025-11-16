// Function: sub_1E44B20
// Address: 0x1e44b20
//
void __fastcall sub_1E44B20(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // rax
  __int64 v7; // r13
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // r14
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rcx
  unsigned __int64 v13; // r15
  __int64 v14; // rdi

  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  v3 = ((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
      | (*(unsigned int *)(a1 + 12) + 2LL)
      | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4;
  v4 = ((v3
       | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | v3
     | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v5 = a2;
  v6 = (v4 | (v4 >> 16) | HIDWORD(v4)) + 1;
  if ( v6 >= a2 )
    v5 = v6;
  if ( v5 > 0xFFFFFFFF )
    v5 = 0xFFFFFFFFLL;
  v7 = malloc(96 * v5);
  if ( !v7 )
    sub_16BD1C0("Allocation failed", 1u);
  v8 = *(_QWORD *)a1;
  v9 = *(_QWORD *)a1 + 96LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v9 )
  {
    v10 = v7;
    do
    {
      if ( v10 )
      {
        *(_DWORD *)(v10 + 24) = 0;
        *(_QWORD *)(v10 + 8) = 0;
        *(_DWORD *)(v10 + 16) = 0;
        *(_DWORD *)(v10 + 20) = 0;
        *(_QWORD *)v10 = 1;
        v11 = *(_QWORD *)(v8 + 8);
        ++*(_QWORD *)v8;
        v12 = *(_QWORD *)(v10 + 8);
        *(_QWORD *)(v10 + 8) = v11;
        LODWORD(v11) = *(_DWORD *)(v8 + 16);
        *(_QWORD *)(v8 + 8) = v12;
        LODWORD(v12) = *(_DWORD *)(v10 + 16);
        *(_DWORD *)(v10 + 16) = v11;
        LODWORD(v11) = *(_DWORD *)(v8 + 20);
        *(_DWORD *)(v8 + 16) = v12;
        LODWORD(v12) = *(_DWORD *)(v10 + 20);
        *(_DWORD *)(v10 + 20) = v11;
        LODWORD(v11) = *(_DWORD *)(v8 + 24);
        *(_DWORD *)(v8 + 20) = v12;
        LODWORD(v12) = *(_DWORD *)(v10 + 24);
        *(_DWORD *)(v10 + 24) = v11;
        *(_DWORD *)(v8 + 24) = v12;
        *(_QWORD *)(v10 + 32) = *(_QWORD *)(v8 + 32);
        *(_QWORD *)(v10 + 40) = *(_QWORD *)(v8 + 40);
        *(_QWORD *)(v10 + 48) = *(_QWORD *)(v8 + 48);
        *(_QWORD *)(v8 + 48) = 0;
        *(_QWORD *)(v8 + 40) = 0;
        *(_QWORD *)(v8 + 32) = 0;
        *(_BYTE *)(v10 + 56) = *(_BYTE *)(v8 + 56);
        *(_DWORD *)(v10 + 60) = *(_DWORD *)(v8 + 60);
        *(_DWORD *)(v10 + 64) = *(_DWORD *)(v8 + 64);
        *(_DWORD *)(v10 + 68) = *(_DWORD *)(v8 + 68);
        *(_DWORD *)(v10 + 72) = *(_DWORD *)(v8 + 72);
        *(_QWORD *)(v10 + 80) = *(_QWORD *)(v8 + 80);
        *(_DWORD *)(v10 + 88) = *(_DWORD *)(v8 + 88);
      }
      v8 += 96LL;
      v10 += 96;
    }
    while ( v9 != v8 );
    v13 = *(_QWORD *)a1;
    v9 = *(_QWORD *)a1 + 96LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v9 )
    {
      do
      {
        v14 = *(_QWORD *)(v9 - 64);
        v9 -= 96LL;
        if ( v14 )
          j_j___libc_free_0(v14, *(_QWORD *)(v9 + 48) - v14);
        j___libc_free_0(*(_QWORD *)(v9 + 8));
      }
      while ( v9 != v13 );
      v9 = *(_QWORD *)a1;
    }
  }
  if ( v9 != a1 + 16 )
    _libc_free(v9);
  *(_QWORD *)a1 = v7;
  *(_DWORD *)(a1 + 12) = v5;
}
