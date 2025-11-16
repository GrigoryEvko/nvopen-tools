// Function: sub_23F9450
// Address: 0x23f9450
//
void __fastcall sub_23F9450(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5, unsigned __int64 a6)
{
  __int64 v6; // rcx
  __int64 *v8; // r14
  __int64 v9; // rax
  __int64 *v10; // r12
  __int64 *i; // rax
  __int64 v12; // rdi
  unsigned int v13; // ecx
  __int64 v14; // rsi
  __int64 *v15; // r12
  unsigned __int64 v16; // r13
  __int64 v17; // rsi
  __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // r8
  __int64 v24; // r13
  __int64 v25; // r13
  __int64 v26; // r12
  _QWORD *v27; // rdi
  __int64 v28; // r13
  unsigned __int64 v29; // r8
  __int64 v30; // r13
  __int64 v31; // r12
  _QWORD *v32; // rdi

  v6 = *(unsigned int *)(a1 + 376);
  if ( (_DWORD)v6 )
    sub_EDA800(a1 + 280, (char *)sub_ED5FB0, 0, v6, a5, a6);
  v8 = *(__int64 **)(a1 + 200);
  v9 = *(unsigned int *)(a1 + 208);
  *(_QWORD *)(a1 + 176) = 0;
  v10 = &v8[v9];
  if ( v8 != v10 )
  {
    for ( i = v8; ; i = *(__int64 **)(a1 + 200) )
    {
      v12 = *v8;
      v13 = (unsigned int)(v8 - i) >> 7;
      v14 = 4096LL << v13;
      if ( v13 >= 0x1E )
        v14 = 0x40000000000LL;
      ++v8;
      sub_C7D6A0(v12, v14, 16);
      if ( v10 == v8 )
        break;
    }
  }
  v15 = *(__int64 **)(a1 + 248);
  v16 = (unsigned __int64)&v15[2 * *(unsigned int *)(a1 + 256)];
  if ( v15 != (__int64 *)v16 )
  {
    do
    {
      v17 = v15[1];
      v18 = *v15;
      v15 += 2;
      sub_C7D6A0(v18, v17, 16);
    }
    while ( (__int64 *)v16 != v15 );
    v16 = *(_QWORD *)(a1 + 248);
  }
  if ( v16 != a1 + 264 )
    _libc_free(v16);
  v19 = *(_QWORD *)(a1 + 200);
  if ( v19 != a1 + 216 )
    _libc_free(v19);
  v20 = *(_QWORD *)(a1 + 152);
  if ( v20 )
    j_j___libc_free_0(v20);
  sub_C7D6A0(*(_QWORD *)(a1 + 128), 16LL * *(unsigned int *)(a1 + 144), 8);
  v21 = *(_QWORD *)(a1 + 96);
  if ( v21 )
    j_j___libc_free_0(v21);
  v22 = *(_QWORD *)(a1 + 72);
  if ( v22 )
    j_j___libc_free_0(v22);
  v23 = *(_QWORD *)(a1 + 48);
  if ( *(_DWORD *)(a1 + 60) )
  {
    v24 = *(unsigned int *)(a1 + 56);
    if ( (_DWORD)v24 )
    {
      v25 = 8 * v24;
      v26 = 0;
      do
      {
        v27 = *(_QWORD **)(v23 + v26);
        if ( v27 != (_QWORD *)-8LL && v27 )
        {
          sub_C7D6A0((__int64)v27, *v27 + 9LL, 8);
          v23 = *(_QWORD *)(a1 + 48);
        }
        v26 += 8;
      }
      while ( v25 != v26 );
    }
  }
  _libc_free(v23);
  if ( *(_DWORD *)(a1 + 36) )
  {
    v28 = *(unsigned int *)(a1 + 32);
    v29 = *(_QWORD *)(a1 + 24);
    if ( (_DWORD)v28 )
    {
      v30 = 8 * v28;
      v31 = 0;
      do
      {
        v32 = *(_QWORD **)(v29 + v31);
        if ( v32 != (_QWORD *)-8LL )
        {
          if ( v32 )
          {
            sub_C7D6A0((__int64)v32, *v32 + 9LL, 8);
            v29 = *(_QWORD *)(a1 + 24);
          }
        }
        v31 += 8;
      }
      while ( v31 != v30 );
    }
    _libc_free(v29);
  }
  else
  {
    _libc_free(*(_QWORD *)(a1 + 24));
  }
}
