// Function: sub_29E0080
// Address: 0x29e0080
//
void __fastcall sub_29E0080(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  int v8; // eax
  __int64 v9; // rcx
  __int64 v10; // rdx
  _QWORD *v11; // rdi
  __int64 v12; // r15
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // r12
  _QWORD *v19; // rdi
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // r12
  __int64 v22; // r13
  __int64 v23; // rdx
  __int64 v24; // rdx
  unsigned __int64 v25; // r13
  unsigned __int64 v26; // rdi
  int v27; // r13d
  unsigned __int64 v28; // [rsp+8h] [rbp-48h]
  unsigned __int64 v29[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = *(unsigned int *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v7 )
  {
    v12 = a1 + 16;
    v16 = sub_C8D7D0(a1, a1 + 16, 0, 0x58u, v29, a6);
    v17 = *(unsigned int *)(a1 + 8);
    v18 = 88 * v17;
    v19 = (_QWORD *)(88 * v17 + v16);
    if ( v19 )
    {
      *v19 = *(_QWORD *)a2;
      v19[1] = v19 + 3;
      v19[2] = 0x800000000LL;
      v13 = *(unsigned int *)(a2 + 16);
      if ( (_DWORD)v13 )
        sub_29DFF10((__int64)(v19 + 1), (char **)(a2 + 8), 5 * v17, v13, v14, v15);
      v18 = 88LL * *(unsigned int *)(a1 + 8);
    }
    v20 = *(_QWORD *)a1;
    v21 = *(_QWORD *)a1 + v18;
    if ( *(_QWORD *)a1 != v21 )
    {
      v22 = v16;
      do
      {
        if ( v22 )
        {
          v23 = *(_QWORD *)v20;
          *(_DWORD *)(v22 + 16) = 0;
          *(_DWORD *)(v22 + 20) = 8;
          *(_QWORD *)v22 = v23;
          *(_QWORD *)(v22 + 8) = v22 + 24;
          v24 = *(unsigned int *)(v20 + 16);
          if ( (_DWORD)v24 )
          {
            v28 = v20;
            sub_29DFF10(v22 + 8, (char **)(v20 + 8), v24, v13, v14, v15);
            v20 = v28;
          }
        }
        v20 += 88LL;
        v22 += 88;
      }
      while ( v21 != v20 );
      v25 = *(_QWORD *)a1;
      v21 = *(_QWORD *)a1 + 88LL * *(unsigned int *)(a1 + 8);
      if ( *(_QWORD *)a1 != v21 )
      {
        do
        {
          v21 -= 88LL;
          v26 = *(_QWORD *)(v21 + 8);
          if ( v26 != v21 + 24 )
            _libc_free(v26);
        }
        while ( v21 != v25 );
        v21 = *(_QWORD *)a1;
      }
    }
    v27 = v29[0];
    if ( v12 != v21 )
      _libc_free(v21);
    *(_QWORD *)a1 = v16;
    *(_DWORD *)(a1 + 12) = v27;
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    v8 = *(_DWORD *)(a1 + 8);
    v9 = 11 * v7;
    v10 = *(_QWORD *)a1;
    v11 = (_QWORD *)(*(_QWORD *)a1 + 8 * v9);
    if ( v11 )
    {
      *v11 = *(_QWORD *)a2;
      v11[1] = v11 + 3;
      v11[2] = 0x800000000LL;
      if ( *(_DWORD *)(a2 + 16) )
        sub_29DFF10((__int64)(v11 + 1), (char **)(a2 + 8), v10, v9, a5, a6);
      v8 = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = v8 + 1;
  }
}
