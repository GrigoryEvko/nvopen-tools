// Function: sub_2BB7B30
// Address: 0x2bb7b30
//
void __fastcall sub_2BB7B30(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rcx
  unsigned __int64 v12; // rdi
  __int64 v13; // rcx
  int v14; // r14d
  unsigned __int64 v15[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x10u, v15, a6);
  v9 = *(_QWORD *)a1;
  v10 = v8;
  v11 = 16LL * *(unsigned int *)(a1 + 8);
  v12 = *(_QWORD *)a1 + v11;
  if ( *(_QWORD *)a1 != v12 )
  {
    v13 = v8 + v11;
    do
    {
      if ( v8 )
      {
        *(_DWORD *)v8 = *(_DWORD *)v9;
        *(_DWORD *)(v8 + 4) = *(_DWORD *)(v9 + 4);
        *(_QWORD *)(v8 + 8) = *(_QWORD *)(v9 + 8);
      }
      v8 += 16;
      v9 += 16LL;
    }
    while ( v8 != v13 );
    v12 = *(_QWORD *)a1;
  }
  v14 = v15[0];
  if ( v6 != v12 )
    _libc_free(v12);
  *(_QWORD *)a1 = v10;
  *(_DWORD *)(a1 + 12) = v14;
}
