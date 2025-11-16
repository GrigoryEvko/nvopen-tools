// Function: sub_2BDE4F0
// Address: 0x2bde4f0
//
void __fastcall sub_2BDE4F0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r14
  __int64 v8; // rsi
  _QWORD *v10; // rax
  __int64 *v11; // rdx
  _QWORD *v12; // r13
  __int64 v13; // rcx
  unsigned __int64 v14; // r12
  _QWORD *v15; // rcx
  __int64 *v16; // r15
  __int64 v17; // rdi
  int v18; // r15d
  unsigned __int64 v19[7]; // [rsp+8h] [rbp-38h] BYREF

  v7 = a1 + 16;
  v8 = a1 + 16;
  v10 = (_QWORD *)sub_C8D7D0(a1, a1 + 16, a2, 8u, v19, a6);
  v11 = *(__int64 **)a1;
  v12 = v10;
  v13 = *(unsigned int *)(a1 + 8);
  v14 = *(_QWORD *)a1 + v13 * 8;
  if ( *(_QWORD *)a1 != v14 )
  {
    v15 = &v10[v13];
    do
    {
      if ( v10 )
      {
        v8 = *v11;
        *v10 = *v11;
        *v11 = 0;
      }
      ++v10;
      ++v11;
    }
    while ( v10 != v15 );
    v16 = *(__int64 **)a1;
    v14 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v14 )
    {
      do
      {
        v17 = *(_QWORD *)(v14 - 8);
        v14 -= 8LL;
        if ( v17 )
          (*(void (__fastcall **)(__int64, __int64, __int64 *))(*(_QWORD *)v17 + 8LL))(v17, v8, v11);
      }
      while ( v16 != (__int64 *)v14 );
      v14 = *(_QWORD *)a1;
    }
  }
  v18 = v19[0];
  if ( v7 != v14 )
    _libc_free(v14);
  *(_QWORD *)a1 = v12;
  *(_DWORD *)(a1 + 12) = v18;
}
