// Function: sub_1DB9560
// Address: 0x1db9560
//
void *__fastcall sub_1DB9560(__int64 a1)
{
  __int64 v2; // r13
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  __int64 v5; // rax
  _QWORD *v6; // rbx
  _QWORD *v7; // r14
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 *v13; // rbx
  unsigned __int64 *v14; // r13
  unsigned __int64 v15; // rdi
  unsigned __int64 *v16; // rbx
  unsigned __int64 v17; // r13
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi

  v2 = *(_QWORD *)(a1 + 288);
  *(_QWORD *)a1 = &unk_49FAD08;
  if ( v2 )
  {
    v3 = *(_QWORD *)(v2 + 136);
    if ( v3 != v2 + 152 )
      _libc_free(v3);
    v4 = *(_QWORD *)(v2 + 96);
    if ( v4 != v2 + 112 )
      _libc_free(v4);
    v5 = *(unsigned int *)(v2 + 88);
    if ( (_DWORD)v5 )
    {
      v6 = *(_QWORD **)(v2 + 72);
      v7 = &v6[7 * v5];
      do
      {
        if ( *v6 != -8 && *v6 != -16 )
        {
          _libc_free(v6[4]);
          _libc_free(v6[1]);
        }
        v6 += 7;
      }
      while ( v7 != v6 );
    }
    j___libc_free_0(*(_QWORD *)(v2 + 72));
    _libc_free(*(_QWORD *)(v2 + 40));
    j_j___libc_free_0(v2, 664);
  }
  v8 = *(_QWORD *)(a1 + 672);
  if ( v8 != a1 + 688 )
    _libc_free(v8);
  v9 = *(_QWORD *)(a1 + 592);
  if ( v9 != a1 + 608 )
    _libc_free(v9);
  v10 = *(_QWORD *)(a1 + 512);
  if ( v10 != a1 + 528 )
    _libc_free(v10);
  v11 = *(_QWORD *)(a1 + 432);
  if ( v11 != a1 + 448 )
    _libc_free(v11);
  v12 = *(_QWORD *)(a1 + 400);
  if ( v12 != a1 + 416 )
    _libc_free(v12);
  v13 = *(unsigned __int64 **)(a1 + 312);
  v14 = &v13[*(unsigned int *)(a1 + 320)];
  while ( v14 != v13 )
  {
    v15 = *v13++;
    _libc_free(v15);
  }
  v16 = *(unsigned __int64 **)(a1 + 360);
  v17 = (unsigned __int64)&v16[2 * *(unsigned int *)(a1 + 368)];
  if ( v16 != (unsigned __int64 *)v17 )
  {
    do
    {
      v18 = *v16;
      v16 += 2;
      _libc_free(v18);
    }
    while ( v16 != (unsigned __int64 *)v17 );
    v17 = *(_QWORD *)(a1 + 360);
  }
  if ( v17 != a1 + 376 )
    _libc_free(v17);
  v19 = *(_QWORD *)(a1 + 312);
  if ( v19 != a1 + 328 )
    _libc_free(v19);
  _libc_free(*(_QWORD *)(a1 + 208));
  _libc_free(*(_QWORD *)(a1 + 184));
  _libc_free(*(_QWORD *)(a1 + 160));
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0((_QWORD *)a1);
}
