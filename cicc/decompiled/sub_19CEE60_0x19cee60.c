// Function: sub_19CEE60
// Address: 0x19cee60
//
void __fastcall sub_19CEE60(__int64 a1)
{
  unsigned __int64 v2; // rdx
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // r13
  int v5; // r8d
  int v6; // r9d
  __int64 v7; // r15
  __int64 v8; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // rcx
  unsigned __int64 v11; // r12
  __int64 v12; // rbx
  int v13; // edx
  __int64 v14; // rdx
  unsigned __int64 v15; // rbx
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // [rsp+8h] [rbp-38h]

  v2 = ((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2;
  v3 = ((((v2 | (*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4)
       | v2
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | ((v2 | (*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4)
     | v2
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v4 = (v3 | (v3 >> 16) | HIDWORD(v3)) + 1;
  if ( v4 > 0xFFFFFFFF )
    v4 = 0xFFFFFFFFLL;
  v7 = malloc(176 * v4);
  if ( !v7 )
    sub_16BD1C0("Allocation failed", 1u);
  v8 = *(unsigned int *)(a1 + 8);
  v9 = *(_QWORD *)a1;
  v10 = 5 * v8;
  v11 = *(_QWORD *)a1 + 176 * v8;
  if ( *(_QWORD *)a1 != v11 )
  {
    v12 = v7;
    do
    {
      if ( v12 )
      {
        *(_QWORD *)v12 = *(_QWORD *)v9;
        *(_QWORD *)(v12 + 8) = *(_QWORD *)(v9 + 8);
        *(_QWORD *)(v12 + 16) = *(_QWORD *)(v9 + 16);
        v13 = *(_DWORD *)(v9 + 24);
        *(_DWORD *)(v12 + 40) = 0;
        *(_DWORD *)(v12 + 24) = v13;
        *(_QWORD *)(v12 + 32) = v12 + 48;
        *(_DWORD *)(v12 + 44) = 16;
        v14 = *(unsigned int *)(v9 + 40);
        if ( (_DWORD)v14 )
        {
          v17 = v9;
          sub_19CEB30(v12 + 32, (char **)(v9 + 32), v14, v10, v5, v6);
          v9 = v17;
        }
      }
      v9 += 176LL;
      v12 += 176;
    }
    while ( v11 != v9 );
    v15 = *(_QWORD *)a1;
    v11 = *(_QWORD *)a1 + 176LL * *(unsigned int *)(a1 + 8);
    if ( v11 != *(_QWORD *)a1 )
    {
      do
      {
        v11 -= 176LL;
        v16 = *(_QWORD *)(v11 + 32);
        if ( v16 != v11 + 48 )
          _libc_free(v16);
      }
      while ( v11 != v15 );
      v11 = *(_QWORD *)a1;
    }
  }
  if ( v11 != a1 + 16 )
    _libc_free(v11);
  *(_QWORD *)a1 = v7;
  *(_DWORD *)(a1 + 12) = v4;
}
