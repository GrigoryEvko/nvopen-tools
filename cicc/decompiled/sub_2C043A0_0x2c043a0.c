// Function: sub_2C043A0
// Address: 0x2c043a0
//
void __fastcall sub_2C043A0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  _QWORD *v8; // rax
  _QWORD *v9; // rdx
  _QWORD *v10; // r13
  __int64 v11; // rcx
  unsigned __int64 v12; // r12
  _QWORD *v13; // rcx
  _QWORD *v14; // r15
  unsigned __int64 v15; // r8
  unsigned __int64 v16; // rdi
  int v17; // r15d
  unsigned __int64 v18; // [rsp+8h] [rbp-48h]
  unsigned __int64 v19[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v8 = (_QWORD *)sub_C8D7D0(a1, a1 + 16, a2, 8u, v19, a6);
  v9 = *(_QWORD **)a1;
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
        *v8 = *v9;
        *v9 = 0;
      }
      ++v8;
      ++v9;
    }
    while ( v8 != v13 );
    v14 = *(_QWORD **)a1;
    v12 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v12 )
    {
      do
      {
        v15 = *(_QWORD *)(v12 - 8);
        v12 -= 8LL;
        if ( v15 )
        {
          v16 = *(_QWORD *)(v15 + 24);
          if ( v16 != v15 + 40 )
          {
            v18 = v15;
            _libc_free(v16);
            v15 = v18;
          }
          j_j___libc_free_0(v15);
        }
      }
      while ( v14 != (_QWORD *)v12 );
      v12 = *(_QWORD *)a1;
    }
  }
  v17 = v19[0];
  if ( v6 != v12 )
    _libc_free(v12);
  *(_QWORD *)a1 = v10;
  *(_DWORD *)(a1 + 12) = v17;
}
