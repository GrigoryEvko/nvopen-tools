// Function: sub_1A7F1E0
// Address: 0x1a7f1e0
//
void __fastcall sub_1A7F1E0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // r13
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // r15
  __int64 v11; // rcx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // r12
  __int64 v14; // rbx
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rdi
  unsigned __int64 v18; // rbx
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // [rsp+8h] [rbp-38h]

  v3 = a2;
  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  v4 = ((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
      | (*(unsigned int *)(a1 + 12) + 2LL)
      | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4;
  v5 = ((v4
       | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | v4
     | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v6 = (v5 | (v5 >> 16) | HIDWORD(v5)) + 1;
  if ( v6 >= a2 )
    v3 = v6;
  v7 = v3;
  if ( v3 > 0xFFFFFFFF )
    v7 = 0xFFFFFFFFLL;
  v10 = malloc(56 * v7);
  if ( !v10 )
    sub_16BD1C0("Allocation failed", 1u);
  v11 = *(unsigned int *)(a1 + 8);
  v12 = *(_QWORD *)a1;
  v13 = *(_QWORD *)a1 + 56 * v11;
  if ( *(_QWORD *)a1 != v13 )
  {
    v14 = v10;
    do
    {
      while ( 1 )
      {
        if ( v14 )
        {
          v15 = *(_QWORD *)v12;
          *(_DWORD *)(v14 + 16) = 0;
          *(_DWORD *)(v14 + 20) = 2;
          *(_QWORD *)v14 = v15;
          *(_QWORD *)(v14 + 8) = v14 + 24;
          v16 = *(unsigned int *)(v12 + 16);
          if ( (_DWORD)v16 )
            break;
        }
        v12 += 56LL;
        v14 += 56;
        if ( v13 == v12 )
          goto LABEL_15;
      }
      v17 = v14 + 8;
      v20 = v12;
      v14 += 56;
      sub_1A7EAB0(v17, v12 + 8, v16, v11, v8, v9);
      v12 = v20 + 56;
    }
    while ( v13 != v20 + 56 );
LABEL_15:
    v18 = *(_QWORD *)a1;
    v13 = *(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v13 )
    {
      do
      {
        v13 -= 56LL;
        v19 = *(_QWORD *)(v13 + 8);
        if ( v19 != v13 + 24 )
          _libc_free(v19);
      }
      while ( v13 != v18 );
      v13 = *(_QWORD *)a1;
    }
  }
  if ( v13 != a1 + 16 )
    _libc_free(v13);
  *(_QWORD *)a1 = v10;
  *(_DWORD *)(a1 + 12) = v7;
}
