// Function: sub_14139D0
// Address: 0x14139d0
//
__int64 __fastcall sub_14139D0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 v5; // rbx
  unsigned __int64 *v6; // rbx
  unsigned __int64 *v7; // r13
  unsigned __int64 v8; // rdi
  unsigned __int64 *v9; // rbx
  unsigned __int64 v10; // r13
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  __int64 v13; // rax
  _QWORD *v14; // rbx
  _QWORD *v15; // r13
  unsigned __int64 v16; // rdi
  __int64 v17; // rax
  _QWORD *v18; // rbx
  _QWORD *v19; // r13
  unsigned __int64 v20; // rdi
  __int64 v21; // rax
  _QWORD *v22; // rbx
  _QWORD *v23; // r13
  __int64 v24; // rdi
  __int64 v25; // rax
  _QWORD *v26; // rbx
  _QWORD *v27; // r13
  unsigned __int64 v28; // rdi
  __int64 v29; // rax
  _QWORD *v30; // rbx
  _QWORD *v31; // r13
  __int64 v32; // rdi
  __int64 v33; // rax
  _QWORD *v34; // rbx
  _QWORD *v35; // r13
  unsigned __int64 v36; // rdi

  v2 = *(_QWORD *)(a1 + 848);
  if ( v2 != *(_QWORD *)(a1 + 840) )
    _libc_free(v2);
  if ( (*(_BYTE *)(a1 + 472) & 1) != 0 )
  {
    v4 = a1 + 480;
    v5 = a1 + 832;
  }
  else
  {
    v3 = *(unsigned int *)(a1 + 488);
    v4 = *(_QWORD *)(a1 + 480);
    if ( !(_DWORD)v3 )
    {
LABEL_65:
      j___libc_free_0(v4);
      goto LABEL_12;
    }
    v5 = v4 + 88 * v3;
  }
  do
  {
    if ( *(_QWORD *)v4 != -8 && *(_QWORD *)v4 != -16 && (*(_BYTE *)(v4 + 16) & 1) == 0 )
      j___libc_free_0(*(_QWORD *)(v4 + 24));
    v4 += 88;
  }
  while ( v4 != v5 );
  if ( (*(_BYTE *)(a1 + 472) & 1) == 0 )
  {
    v4 = *(_QWORD *)(a1 + 480);
    goto LABEL_65;
  }
LABEL_12:
  v6 = *(unsigned __int64 **)(a1 + 376);
  v7 = &v6[*(unsigned int *)(a1 + 384)];
  while ( v7 != v6 )
  {
    v8 = *v6++;
    _libc_free(v8);
  }
  v9 = *(unsigned __int64 **)(a1 + 424);
  v10 = (unsigned __int64)&v9[2 * *(unsigned int *)(a1 + 432)];
  if ( v9 != (unsigned __int64 *)v10 )
  {
    do
    {
      v11 = *v9;
      v9 += 2;
      _libc_free(v11);
    }
    while ( (unsigned __int64 *)v10 != v9 );
    v10 = *(_QWORD *)(a1 + 424);
  }
  if ( v10 != a1 + 440 )
    _libc_free(v10);
  v12 = *(_QWORD *)(a1 + 376);
  if ( v12 != a1 + 392 )
    _libc_free(v12);
  j___libc_free_0(*(_QWORD *)(a1 + 336));
  j___libc_free_0(*(_QWORD *)(a1 + 304));
  v13 = *(unsigned int *)(a1 + 248);
  if ( (_DWORD)v13 )
  {
    v14 = *(_QWORD **)(a1 + 232);
    v15 = &v14[10 * v13];
    do
    {
      if ( *v14 != -16 && *v14 != -8 )
      {
        v16 = v14[3];
        if ( v16 != v14[2] )
          _libc_free(v16);
      }
      v14 += 10;
    }
    while ( v15 != v14 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 232));
  v17 = *(unsigned int *)(a1 + 216);
  if ( (_DWORD)v17 )
  {
    v18 = *(_QWORD **)(a1 + 200);
    v19 = &v18[10 * v17];
    do
    {
      if ( *v18 != -16 && *v18 != -8 )
      {
        v20 = v18[3];
        if ( v20 != v18[2] )
          _libc_free(v20);
      }
      v18 += 10;
    }
    while ( v19 != v18 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 200));
  v21 = *(unsigned int *)(a1 + 184);
  if ( (_DWORD)v21 )
  {
    v22 = *(_QWORD **)(a1 + 168);
    v23 = &v22[5 * v21];
    do
    {
      if ( *v22 != -8 && *v22 != -16 )
      {
        v24 = v22[1];
        if ( v24 )
          j_j___libc_free_0(v24, v22[3] - v24);
      }
      v22 += 5;
    }
    while ( v23 != v22 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 168));
  v25 = *(unsigned int *)(a1 + 152);
  if ( (_DWORD)v25 )
  {
    v26 = *(_QWORD **)(a1 + 136);
    v27 = &v26[10 * v25];
    do
    {
      if ( *v26 != -16 && *v26 != -8 )
      {
        v28 = v26[3];
        if ( v28 != v26[2] )
          _libc_free(v28);
      }
      v26 += 10;
    }
    while ( v27 != v26 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 136));
  v29 = *(unsigned int *)(a1 + 120);
  if ( (_DWORD)v29 )
  {
    v30 = *(_QWORD **)(a1 + 104);
    v31 = &v30[9 * v29];
    do
    {
      if ( *v30 != -16 && *v30 != -4 )
      {
        v32 = v30[2];
        if ( v32 )
          j_j___libc_free_0(v32, v30[4] - v32);
      }
      v30 += 9;
    }
    while ( v31 != v30 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 104));
  v33 = *(unsigned int *)(a1 + 88);
  if ( (_DWORD)v33 )
  {
    v34 = *(_QWORD **)(a1 + 72);
    v35 = &v34[10 * v33];
    do
    {
      if ( *v34 != -16 && *v34 != -8 )
      {
        v36 = v34[3];
        if ( v36 != v34[2] )
          _libc_free(v36);
      }
      v34 += 10;
    }
    while ( v35 != v34 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 72));
  j___libc_free_0(*(_QWORD *)(a1 + 40));
  return j___libc_free_0(*(_QWORD *)(a1 + 8));
}
