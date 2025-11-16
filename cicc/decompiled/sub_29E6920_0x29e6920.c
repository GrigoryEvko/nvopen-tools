// Function: sub_29E6920
// Address: 0x29e6920
//
void __fastcall sub_29E6920(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r14
  __int64 v8; // rsi
  _QWORD *v10; // rax
  __int64 v11; // r8
  __int64 *v12; // rdx
  _QWORD *v13; // r13
  __int64 v14; // rcx
  unsigned __int64 v15; // r12
  __int64 v16; // rcx
  __int64 *v17; // r15
  __int64 v18; // rdi
  int v19; // r15d
  unsigned __int64 v20[7]; // [rsp+8h] [rbp-38h] BYREF

  v7 = a1 + 16;
  v8 = a1 + 16;
  v10 = (_QWORD *)sub_C8D7D0(a1, a1 + 16, a2, 8u, v20, a6);
  v12 = *(__int64 **)a1;
  v13 = v10;
  v14 = *(unsigned int *)(a1 + 8);
  v15 = *(_QWORD *)a1 + v14 * 8;
  if ( *(_QWORD *)a1 != v15 )
  {
    v16 = (__int64)&v10[v14];
    do
    {
      if ( v10 )
      {
        v8 = *v12;
        *v10 = *v12;
        *v12 = 0;
      }
      ++v10;
      ++v12;
    }
    while ( v10 != (_QWORD *)v16 );
    v17 = *(__int64 **)a1;
    v15 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v15 )
    {
      do
      {
        v18 = *(_QWORD *)(v15 - 8);
        v15 -= 8LL;
        if ( v18 )
          sub_BA65D0(v18, v8, (__int64)v12, v16, v11);
      }
      while ( v17 != (__int64 *)v15 );
      v15 = *(_QWORD *)a1;
    }
  }
  v19 = v20[0];
  if ( v7 != v15 )
    _libc_free(v15);
  *(_QWORD *)a1 = v13;
  *(_DWORD *)(a1 + 12) = v19;
}
