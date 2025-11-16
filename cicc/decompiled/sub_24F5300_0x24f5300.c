// Function: sub_24F5300
// Address: 0x24f5300
//
void __fastcall sub_24F5300(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r14
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // r12
  __int64 v14; // rbx
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rdi
  unsigned __int64 v18; // rbx
  unsigned __int64 v19; // rdi
  int v20; // ebx
  unsigned __int64 v21; // [rsp+8h] [rbp-48h]
  unsigned __int64 v22[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v11 = sub_C8D7D0(a1, a1 + 16, a2, 0x28u, v22, a6);
  v12 = *(_QWORD *)a1;
  v13 = *(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v13 )
  {
    v14 = v11;
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
        v12 += 40LL;
        v14 += 40;
        if ( v13 == v12 )
          goto LABEL_7;
      }
      v17 = v14 + 8;
      v21 = v12;
      v14 += 40;
      sub_24F4DA0(v17, (char **)(v12 + 8), v16, v8, v9, v10);
      v12 = v21 + 40;
    }
    while ( v13 != v21 + 40 );
LABEL_7:
    v18 = *(_QWORD *)a1;
    v13 = *(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v13 )
    {
      do
      {
        v13 -= 40LL;
        v19 = *(_QWORD *)(v13 + 8);
        if ( v19 != v13 + 24 )
          _libc_free(v19);
      }
      while ( v13 != v18 );
      v13 = *(_QWORD *)a1;
    }
  }
  v20 = v22[0];
  if ( v6 != v13 )
    _libc_free(v13);
  *(_QWORD *)a1 = v11;
  *(_DWORD *)(a1 + 12) = v20;
}
