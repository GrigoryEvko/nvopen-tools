// Function: sub_29D07A0
// Address: 0x29d07a0
//
void __fastcall sub_29D07A0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rsi
  _QWORD *v8; // rax
  __int64 *v9; // rdx
  _QWORD *v10; // r13
  __int64 v11; // rcx
  unsigned __int64 v12; // r12
  _QWORD *v13; // rcx
  __int64 *v14; // r14
  __int64 v15; // r15
  __int64 v16; // rdx
  __int64 v17; // rcx
  int v18; // r15d
  unsigned __int64 v19[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = a1 + 16;
  v8 = (_QWORD *)sub_C8D7D0(a1, a1 + 16, a2, 8u, v19, a6);
  v9 = *(__int64 **)a1;
  v10 = v8;
  v11 = *(unsigned int *)(a1 + 8);
  v12 = *(_QWORD *)a1 + v11 * 8;
  if ( *(_QWORD *)a1 != v12 )
  {
    v13 = &v8[v11];
    do
    {
      if ( v8 )
      {
        v7 = *v9;
        *v8 = *v9;
        *v9 = 0;
      }
      ++v8;
      ++v9;
    }
    while ( v8 != v13 );
    v14 = *(__int64 **)a1;
    v12 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v12 )
    {
      do
      {
        v15 = *(_QWORD *)(v12 - 8);
        v12 -= 8LL;
        if ( v15 )
        {
          sub_B30220(v15);
          *(_DWORD *)(v15 + 4) = *(_DWORD *)(v15 + 4) & 0xF8000000 | 1;
          sub_B2F9E0(v15, v7, v16, v17);
          sub_BD2DD0(v15);
        }
      }
      while ( v14 != (__int64 *)v12 );
      v12 = *(_QWORD *)a1;
    }
  }
  v18 = v19[0];
  if ( a1 + 16 != v12 )
    _libc_free(v12);
  *(_QWORD *)a1 = v10;
  *(_DWORD *)(a1 + 12) = v18;
}
