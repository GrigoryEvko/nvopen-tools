// Function: sub_322DF10
// Address: 0x322df10
//
void __fastcall sub_322DF10(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v8; // r12
  _QWORD *v9; // rax
  __int64 v10; // rcx
  unsigned __int64 v11; // r14
  _QWORD *v12; // rcx
  _QWORD *v13; // rdx
  _QWORD *v14; // r15
  unsigned __int64 v15; // rdi
  int v16; // r15d
  unsigned __int64 v17[7]; // [rsp+8h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x20u, v17, a6);
  v9 = *(_QWORD **)a1;
  v10 = 32LL * *(unsigned int *)(a1 + 8);
  v11 = *(_QWORD *)a1 + v10;
  if ( *(_QWORD *)a1 != v11 )
  {
    v12 = (_QWORD *)(v8 + v10);
    v13 = (_QWORD *)v8;
    do
    {
      if ( v13 )
      {
        *v13 = *v9;
        v13[1] = v9[1];
        v13[2] = v9[2];
        v13[3] = v9[3];
        v9[3] = 0;
        v9[2] = 0;
        v9[1] = 0;
      }
      v13 += 4;
      v9 += 4;
    }
    while ( v13 != v12 );
    v14 = *(_QWORD **)a1;
    v11 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v11 )
    {
      do
      {
        v15 = *(_QWORD *)(v11 - 24);
        v11 -= 32LL;
        if ( v15 )
          j_j___libc_free_0(v15);
      }
      while ( (_QWORD *)v11 != v14 );
      v11 = *(_QWORD *)a1;
    }
  }
  v16 = v17[0];
  if ( v6 != v11 )
    _libc_free(v11);
  *(_QWORD *)a1 = v8;
  *(_DWORD *)(a1 + 12) = v16;
}
