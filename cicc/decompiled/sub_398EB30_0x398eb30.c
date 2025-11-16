// Function: sub_398EB30
// Address: 0x398eb30
//
void __fastcall sub_398EB30(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rcx
  int v9; // r8d
  int v10; // r9d
  __int64 v11; // r15
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
  v11 = malloc(v7 << 6);
  if ( !v11 )
    sub_16BD1C0("Allocation failed", 1u);
  v12 = *(_QWORD *)a1;
  v13 = *(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 6);
  if ( *(_QWORD *)a1 != v13 )
  {
    v14 = v11;
    do
    {
      while ( 1 )
      {
        if ( v14 )
        {
          *(_QWORD *)v14 = *(_QWORD *)v12;
          v15 = *(_QWORD *)(v12 + 8);
          *(_DWORD *)(v14 + 24) = 0;
          *(_QWORD *)(v14 + 8) = v15;
          *(_QWORD *)(v14 + 16) = v14 + 32;
          *(_DWORD *)(v14 + 28) = 1;
          v16 = *(unsigned int *)(v12 + 24);
          if ( (_DWORD)v16 )
            break;
        }
        v12 += 64LL;
        v14 += 64;
        if ( v13 == v12 )
          goto LABEL_15;
      }
      v17 = v14 + 16;
      v20 = v12;
      v14 += 64;
      sub_39849D0(v17, (char **)(v12 + 16), v16, v8, v9, v10);
      v12 = v20 + 64;
    }
    while ( v13 != v20 + 64 );
LABEL_15:
    v18 = *(_QWORD *)a1;
    v13 = *(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 6);
    if ( *(_QWORD *)a1 != v13 )
    {
      do
      {
        v13 -= 64LL;
        v19 = *(_QWORD *)(v13 + 16);
        if ( v19 != v13 + 32 )
          _libc_free(v19);
      }
      while ( v13 != v18 );
      v13 = *(_QWORD *)a1;
    }
  }
  if ( v13 != a1 + 16 )
    _libc_free(v13);
  *(_QWORD *)a1 = v11;
  *(_DWORD *)(a1 + 12) = v7;
}
