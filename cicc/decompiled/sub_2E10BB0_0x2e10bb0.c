// Function: sub_2E10BB0
// Address: 0x2e10bb0
//
void __fastcall sub_2E10BB0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  __int64 *v7; // r14
  __int64 *v8; // r12
  __int64 i; // rax
  __int64 v10; // rdi
  unsigned int v11; // ecx
  __int64 v12; // rsi
  __int64 *v13; // r12
  unsigned __int64 v14; // r13
  __int64 v15; // rsi
  __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // r12
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  __int64 v21; // rax
  _QWORD *v22; // rbx
  _QWORD *v23; // r13
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi

  sub_2E10910(a1);
  v2 = *(_QWORD *)(a1 + 424);
  if ( v2 != a1 + 440 )
    _libc_free(v2);
  v3 = *(_QWORD *)(a1 + 344);
  if ( v3 != a1 + 360 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 264);
  if ( v4 != a1 + 280 )
    _libc_free(v4);
  v5 = *(_QWORD *)(a1 + 184);
  if ( v5 != a1 + 200 )
    _libc_free(v5);
  v6 = *(_QWORD *)(a1 + 152);
  if ( v6 != a1 + 168 )
    _libc_free(v6);
  v7 = *(__int64 **)(a1 + 72);
  v8 = &v7[*(unsigned int *)(a1 + 80)];
  if ( v7 != v8 )
  {
    for ( i = *(_QWORD *)(a1 + 72); ; i = *(_QWORD *)(a1 + 72) )
    {
      v10 = *v7;
      v11 = (unsigned int)(((__int64)v7 - i) >> 3) >> 7;
      v12 = 4096LL << v11;
      if ( v11 >= 0x1E )
        v12 = 0x40000000000LL;
      ++v7;
      sub_C7D6A0(v10, v12, 16);
      if ( v8 == v7 )
        break;
    }
  }
  v13 = *(__int64 **)(a1 + 120);
  v14 = (unsigned __int64)&v13[2 * *(unsigned int *)(a1 + 128)];
  if ( v13 != (__int64 *)v14 )
  {
    do
    {
      v15 = v13[1];
      v16 = *v13;
      v13 += 2;
      sub_C7D6A0(v16, v15, 16);
    }
    while ( (__int64 *)v14 != v13 );
    v14 = *(_QWORD *)(a1 + 120);
  }
  if ( v14 != a1 + 136 )
    _libc_free(v14);
  v17 = *(_QWORD *)(a1 + 72);
  if ( v17 != a1 + 88 )
    _libc_free(v17);
  v18 = *(_QWORD *)(a1 + 48);
  if ( v18 )
  {
    v19 = *(_QWORD *)(v18 + 184);
    if ( v19 != v18 + 200 )
      _libc_free(v19);
    v20 = *(_QWORD *)(v18 + 144);
    if ( v20 != v18 + 160 )
      _libc_free(v20);
    v21 = *(unsigned int *)(v18 + 136);
    if ( (_DWORD)v21 )
    {
      v22 = *(_QWORD **)(v18 + 120);
      v23 = &v22[19 * v21];
      do
      {
        if ( *v22 != -8192 && *v22 != -4096 )
        {
          v24 = v22[10];
          if ( (_QWORD *)v24 != v22 + 12 )
            _libc_free(v24);
          v25 = v22[1];
          if ( (_QWORD *)v25 != v22 + 3 )
            _libc_free(v25);
        }
        v22 += 19;
      }
      while ( v23 != v22 );
      v21 = *(unsigned int *)(v18 + 136);
    }
    sub_C7D6A0(*(_QWORD *)(v18 + 120), 152 * v21, 8);
    v26 = *(_QWORD *)(v18 + 40);
    if ( v26 != v18 + 56 )
      _libc_free(v26);
    j_j___libc_free_0(v18);
  }
}
