// Function: sub_3502770
// Address: 0x3502770
//
void __fastcall sub_3502770(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
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
  unsigned __int64 v17; // rbx
  unsigned __int64 v18; // rdi
  int v19; // ebx
  unsigned __int64 v20; // [rsp+8h] [rbp-48h]
  unsigned __int64 v21[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v11 = sub_C8D7D0(a1, a1 + 16, a2, 0x70u, v21, a6);
  v12 = *(_QWORD *)a1;
  v13 = *(_QWORD *)a1 + 112LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v13 )
  {
    v14 = v11;
    do
    {
      if ( v14 )
      {
        v15 = *(_QWORD *)v12;
        *(_DWORD *)(v14 + 16) = 0;
        *(_DWORD *)(v14 + 20) = 4;
        *(_QWORD *)v14 = v15;
        *(_QWORD *)(v14 + 8) = v14 + 24;
        v16 = *(unsigned int *)(v12 + 16);
        if ( (_DWORD)v16 )
        {
          v20 = v12;
          sub_35018C0(v14 + 8, (char **)(v12 + 8), v16, v8, v9, v10);
          v12 = v20;
        }
        *(_DWORD *)(v14 + 88) = *(_DWORD *)(v12 + 88);
        *(_QWORD *)(v14 + 96) = *(_QWORD *)(v12 + 96);
        *(_QWORD *)(v14 + 104) = *(_QWORD *)(v12 + 104);
      }
      v12 += 112LL;
      v14 += 112;
    }
    while ( v13 != v12 );
    v17 = *(_QWORD *)a1;
    v13 = *(_QWORD *)a1 + 112LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v13 )
    {
      do
      {
        v13 -= 112LL;
        v18 = *(_QWORD *)(v13 + 8);
        if ( v18 != v13 + 24 )
          _libc_free(v18);
      }
      while ( v13 != v17 );
      v13 = *(_QWORD *)a1;
    }
  }
  v19 = v21[0];
  if ( v6 != v13 )
    _libc_free(v13);
  *(_QWORD *)a1 = v11;
  *(_DWORD *)(a1 + 12) = v19;
}
