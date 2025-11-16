// Function: sub_1E0FED0
// Address: 0x1e0fed0
//
__int64 __fastcall sub_1E0FED0(__int64 a1)
{
  __int64 v2; // rdx
  _QWORD *v3; // rbx
  __int64 v4; // rax
  _QWORD *v5; // r12
  __int64 v6; // rsi
  unsigned __int64 *v7; // rcx
  unsigned __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  void (***v12)(void); // rdi
  void (*v13)(void); // rax
  _QWORD *v14; // rbx
  unsigned __int64 v15; // rdi
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 result; // rax
  _QWORD *v19; // r14
  _QWORD *v20; // rbx
  _QWORD *v21; // r12
  __int64 v22; // rbx
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  __int64 v25; // r13
  unsigned __int64 v26; // r12
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdi

  v2 = *(_QWORD *)(a1 + 360);
  if ( v2 )
    memset(*(void **)(a1 + 352), 0, 8 * v2);
  v3 = *(_QWORD **)(a1 + 328);
  while ( (_QWORD *)(a1 + 320) != v3 )
  {
    v4 = v3[3];
    v5 = v3;
    v3[4] = v3 + 3;
    v6 = (__int64)v3;
    v3[3] = (unsigned __int64)(v3 + 3) | v4 & 7;
    v3 = (_QWORD *)v3[1];
    sub_1DD5B80(a1 + 320, v6);
    v7 = (unsigned __int64 *)v5[1];
    v8 = *v5 & 0xFFFFFFFFFFFFFFF8LL;
    *v7 = v8 | *v7 & 7;
    *(_QWORD *)(v8 + 8) = v7;
    *v5 &= 7uLL;
    v5[1] = 0;
    sub_1E0A230(a1 + 320, v5);
  }
  v9 = *(_QWORD *)(a1 + 96);
  if ( v9 != *(_QWORD *)(a1 + 104) )
    *(_QWORD *)(a1 + 104) = v9;
  *(_QWORD *)(a1 + 224) = 0;
  v10 = *(_QWORD *)(a1 + 496);
  *(_DWORD *)(a1 + 240) = 0;
  *(_QWORD *)(a1 + 312) = 0;
  if ( v10 != *(_QWORD *)(a1 + 504) )
    *(_QWORD *)(a1 + 504) = v10;
  *(_DWORD *)(a1 + 616) = 0;
  v11 = *(_QWORD *)(a1 + 40);
  if ( v11 )
    sub_1E09EE0(v11);
  v12 = *(void (****)(void))(a1 + 48);
  if ( v12 )
  {
    v13 = **v12;
    if ( v13 != nullsub_702 )
      v13();
  }
  v14 = *(_QWORD **)(a1 + 56);
  v15 = v14[14];
  if ( (_QWORD *)v15 != v14 + 16 )
    _libc_free(v15);
  v16 = v14[10];
  if ( v16 )
    j_j___libc_free_0(v16, v14[12] - v16);
  v17 = v14[1];
  if ( v17 )
    j_j___libc_free_0(v17, v14[3] - v17);
  result = sub_1E0FB20(*(_QWORD *)(a1 + 64));
  v19 = *(_QWORD **)(a1 + 72);
  if ( v19 )
  {
    v20 = (_QWORD *)v19[2];
    v21 = (_QWORD *)v19[1];
    if ( v20 != v21 )
    {
      do
      {
        if ( *v21 )
          result = j_j___libc_free_0(*v21, v21[2] - *v21);
        v21 += 3;
      }
      while ( v20 != v21 );
      v21 = (_QWORD *)v19[1];
    }
    if ( v21 )
      result = j_j___libc_free_0(v21, v19[3] - (_QWORD)v21);
  }
  v22 = *(_QWORD *)(a1 + 88);
  if ( v22 )
  {
    v23 = *(_QWORD *)(v22 + 592);
    if ( v23 != v22 + 608 )
      _libc_free(v23);
    v24 = *(_QWORD *)(v22 + 480);
    if ( v24 != v22 + 496 )
      _libc_free(v24);
    v25 = *(_QWORD *)(v22 + 208);
    v26 = v25 + ((unsigned __int64)*(unsigned int *)(v22 + 216) << 6);
    if ( v25 != v26 )
    {
      do
      {
        v26 -= 64LL;
        v27 = *(_QWORD *)(v26 + 16);
        if ( v27 != v26 + 32 )
          _libc_free(v27);
      }
      while ( v25 != v26 );
      v26 = *(_QWORD *)(v22 + 208);
    }
    if ( v26 != v22 + 224 )
      _libc_free(v26);
    v28 = *(_QWORD *)(v22 + 128);
    if ( v28 != v22 + 144 )
      _libc_free(v28);
    j___libc_free_0(*(_QWORD *)(v22 + 104));
    j___libc_free_0(*(_QWORD *)(v22 + 72));
    j___libc_free_0(*(_QWORD *)(v22 + 40));
    return j___libc_free_0(*(_QWORD *)(v22 + 8));
  }
  return result;
}
