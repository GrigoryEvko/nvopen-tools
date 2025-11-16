// Function: sub_2B47650
// Address: 0x2b47650
//
_QWORD *__fastcall sub_2B47650(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  __int64 v9; // rcx
  unsigned __int64 v10; // r12
  _QWORD *v11; // rcx
  unsigned __int64 v12; // r15
  unsigned __int64 *v13; // r13
  unsigned __int64 *v14; // r14
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  __int64 v18; // rax
  int v19; // r13d
  _QWORD *v21; // [rsp+8h] [rbp-58h]
  __int64 v22; // [rsp+10h] [rbp-50h]
  _QWORD *v23; // [rsp+18h] [rbp-48h]
  unsigned __int64 v24[7]; // [rsp+28h] [rbp-38h] BYREF

  v22 = a1 + 16;
  v7 = (_QWORD *)sub_C8D7D0(a1, a1 + 16, a2, 8u, v24, a6);
  v8 = *(_QWORD **)a1;
  v21 = v7;
  v9 = *(unsigned int *)(a1 + 8);
  v10 = *(_QWORD *)a1 + v9 * 8;
  if ( v8 != &v8[v9] )
  {
    v11 = &v7[v9];
    do
    {
      if ( v7 )
      {
        *v7 = *v8;
        *v8 = 0;
      }
      ++v7;
      ++v8;
    }
    while ( v7 != v11 );
    v10 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    v23 = *(_QWORD **)a1;
    if ( *(_QWORD *)a1 != v10 )
    {
      do
      {
        v12 = *(_QWORD *)(v10 - 8);
        v10 -= 8LL;
        if ( v12 )
        {
          v13 = *(unsigned __int64 **)(v12 + 240);
          v14 = &v13[10 * *(unsigned int *)(v12 + 248)];
          if ( v13 != v14 )
          {
            do
            {
              v14 -= 10;
              if ( (unsigned __int64 *)*v14 != v14 + 2 )
                _libc_free(*v14);
            }
            while ( v13 != v14 );
            v14 = *(unsigned __int64 **)(v12 + 240);
          }
          if ( v14 != (unsigned __int64 *)(v12 + 256) )
            _libc_free((unsigned __int64)v14);
          v15 = *(_QWORD *)(v12 + 208);
          if ( v15 != v12 + 224 )
            _libc_free(v15);
          v16 = *(_QWORD *)(v12 + 144);
          if ( v16 != v12 + 160 )
            _libc_free(v16);
          v17 = *(_QWORD *)(v12 + 112);
          if ( v17 != v12 + 128 )
            _libc_free(v17);
          v18 = *(_QWORD *)(v12 + 96);
          if ( v18 != 0 && v18 != -4096 && v18 != -8192 )
            sub_BD60C0((_QWORD *)(v12 + 80));
          if ( *(_QWORD *)v12 != v12 + 16 )
            _libc_free(*(_QWORD *)v12);
          j_j___libc_free_0(v12);
        }
      }
      while ( v23 != (_QWORD *)v10 );
      v10 = *(_QWORD *)a1;
    }
  }
  v19 = v24[0];
  if ( v22 != v10 )
    _libc_free(v10);
  *(_DWORD *)(a1 + 12) = v19;
  *(_QWORD *)a1 = v21;
  return v21;
}
