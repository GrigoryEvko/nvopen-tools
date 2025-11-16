// Function: sub_EDAD10
// Address: 0xedad10
//
__int64 __fastcall sub_EDAD10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5, unsigned __int64 a6)
{
  __int64 v7; // rcx
  __int64 *v8; // r14
  __int64 v9; // rax
  __int64 *v10; // rbx
  __int64 *i; // rax
  __int64 v12; // rdi
  unsigned int v13; // ecx
  __int64 *v14; // rbx
  __int64 *v15; // r13
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // rsi
  __int64 v20; // rdi
  __int64 v21; // rdi
  __int64 v22; // r8
  __int64 v23; // r13
  __int64 v24; // r13
  __int64 v25; // rbx
  _QWORD *v26; // rdi
  __int64 v27; // r13
  __int64 v28; // r8
  __int64 v29; // r13
  __int64 v30; // rbx
  _QWORD *v31; // rdi

  v7 = *(unsigned int *)(a1 + 376);
  if ( (_DWORD)v7 )
  {
    a2 = (__int64)sub_ED5FB0;
    sub_EDA800(a1 + 280, (char *)sub_ED5FB0, 0, v7, a5, a6);
  }
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
      a2 = 4096LL << v13;
      if ( v13 >= 0x1E )
        a2 = 0x40000000000LL;
      ++v8;
      sub_C7D6A0(v12, a2, 16);
      if ( v10 == v8 )
        break;
    }
  }
  v14 = *(__int64 **)(a1 + 248);
  v15 = &v14[2 * *(unsigned int *)(a1 + 256)];
  if ( v14 != v15 )
  {
    do
    {
      a2 = v14[1];
      v16 = *v14;
      v14 += 2;
      sub_C7D6A0(v16, a2, 16);
    }
    while ( v15 != v14 );
    v15 = *(__int64 **)(a1 + 248);
  }
  if ( v15 != (__int64 *)(a1 + 264) )
    _libc_free(v15, a2);
  v17 = *(_QWORD *)(a1 + 200);
  if ( v17 != a1 + 216 )
    _libc_free(v17, a2);
  v18 = *(_QWORD *)(a1 + 152);
  if ( v18 )
    j_j___libc_free_0(v18, *(_QWORD *)(a1 + 168) - v18);
  v19 = 16LL * *(unsigned int *)(a1 + 144);
  sub_C7D6A0(*(_QWORD *)(a1 + 128), v19, 8);
  v20 = *(_QWORD *)(a1 + 96);
  if ( v20 )
  {
    v19 = *(_QWORD *)(a1 + 112) - v20;
    j_j___libc_free_0(v20, v19);
  }
  v21 = *(_QWORD *)(a1 + 72);
  if ( v21 )
  {
    v19 = *(_QWORD *)(a1 + 88) - v21;
    j_j___libc_free_0(v21, v19);
  }
  v22 = *(_QWORD *)(a1 + 48);
  if ( *(_DWORD *)(a1 + 60) )
  {
    v23 = *(unsigned int *)(a1 + 56);
    if ( (_DWORD)v23 )
    {
      v24 = 8 * v23;
      v25 = 0;
      do
      {
        v26 = *(_QWORD **)(v22 + v25);
        if ( v26 != (_QWORD *)-8LL && v26 )
        {
          v19 = *v26 + 9LL;
          sub_C7D6A0((__int64)v26, v19, 8);
          v22 = *(_QWORD *)(a1 + 48);
        }
        v25 += 8;
      }
      while ( v24 != v25 );
    }
  }
  _libc_free(v22, v19);
  if ( *(_DWORD *)(a1 + 36) )
  {
    v27 = *(unsigned int *)(a1 + 32);
    v28 = *(_QWORD *)(a1 + 24);
    if ( (_DWORD)v27 )
    {
      v29 = 8 * v27;
      v30 = 0;
      do
      {
        v31 = *(_QWORD **)(v28 + v30);
        if ( v31 != (_QWORD *)-8LL && v31 )
        {
          v19 = *v31 + 9LL;
          sub_C7D6A0((__int64)v31, v19, 8);
          v28 = *(_QWORD *)(a1 + 24);
        }
        v30 += 8;
      }
      while ( v30 != v29 );
    }
  }
  else
  {
    v28 = *(_QWORD *)(a1 + 24);
  }
  _libc_free(v28, v19);
  return j_j___libc_free_0(a1, 400);
}
