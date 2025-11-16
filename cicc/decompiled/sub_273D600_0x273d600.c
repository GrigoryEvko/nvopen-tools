// Function: sub_273D600
// Address: 0x273d600
//
void __fastcall sub_273D600(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  __int64 v8; // rbx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // r12
  __int64 v11; // rdx
  unsigned __int64 v12; // r15
  unsigned __int64 v13; // rdi
  int v14; // r15d
  unsigned __int64 v15[7]; // [rsp+8h] [rbp-38h] BYREF

  v7 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x18u, v15, a6);
  v9 = *(_QWORD *)a1;
  v10 = *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v10 )
  {
    v11 = v8;
    do
    {
      if ( v11 )
      {
        *(_QWORD *)v11 = *(_QWORD *)v9;
        *(_DWORD *)(v11 + 16) = *(_DWORD *)(v9 + 16);
        *(_QWORD *)(v11 + 8) = *(_QWORD *)(v9 + 8);
        *(_DWORD *)(v9 + 16) = 0;
      }
      v9 += 24LL;
      v11 += 24;
    }
    while ( v10 != v9 );
    v12 = *(_QWORD *)a1;
    v10 = *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v10 )
    {
      do
      {
        v10 -= 24LL;
        if ( *(_DWORD *)(v10 + 16) > 0x40u )
        {
          v13 = *(_QWORD *)(v10 + 8);
          if ( v13 )
            j_j___libc_free_0_0(v13);
        }
      }
      while ( v10 != v12 );
      v10 = *(_QWORD *)a1;
    }
  }
  v14 = v15[0];
  if ( v7 != v10 )
    _libc_free(v10);
  *(_QWORD *)a1 = v8;
  *(_DWORD *)(a1 + 12) = v14;
}
