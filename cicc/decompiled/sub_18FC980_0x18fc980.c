// Function: sub_18FC980
// Address: 0x18fc980
//
__int64 __fastcall sub_18FC980(__int64 a1)
{
  unsigned __int64 *v2; // rbx
  __int64 v3; // rax
  unsigned __int64 *v4; // r13
  unsigned __int64 v5; // rdi
  unsigned __int64 *v6; // rbx
  unsigned __int64 v7; // r13
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  unsigned __int64 *v10; // rbx
  __int64 v11; // rax
  unsigned __int64 *v12; // r13
  unsigned __int64 v13; // rdi
  unsigned __int64 *v14; // rbx
  unsigned __int64 v15; // r13
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 *v18; // rbx
  __int64 v19; // rax
  unsigned __int64 *v20; // r13
  unsigned __int64 v21; // rdi
  unsigned __int64 *v22; // rbx
  unsigned __int64 v23; // r13
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  __int64 result; // rax
  __int64 v27; // r13
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdi
  _QWORD *v30; // rbx
  _QWORD *v31; // r12
  __int64 v32; // rax

  j___libc_free_0(*(_QWORD *)(a1 + 552));
  v2 = *(unsigned __int64 **)(a1 + 456);
  v3 = *(unsigned int *)(a1 + 464);
  *(_QWORD *)(a1 + 432) = 0;
  v4 = &v2[v3];
  while ( v4 != v2 )
  {
    v5 = *v2++;
    _libc_free(v5);
  }
  v6 = *(unsigned __int64 **)(a1 + 504);
  v7 = (unsigned __int64)&v6[2 * *(unsigned int *)(a1 + 512)];
  if ( v6 != (unsigned __int64 *)v7 )
  {
    do
    {
      v8 = *v6;
      v6 += 2;
      _libc_free(v8);
    }
    while ( (unsigned __int64 *)v7 != v6 );
    v7 = *(_QWORD *)(a1 + 504);
  }
  if ( v7 != a1 + 520 )
    _libc_free(v7);
  v9 = *(_QWORD *)(a1 + 456);
  if ( v9 != a1 + 472 )
    _libc_free(v9);
  j___libc_free_0(*(_QWORD *)(a1 + 400));
  v10 = *(unsigned __int64 **)(a1 + 304);
  v11 = *(unsigned int *)(a1 + 312);
  *(_QWORD *)(a1 + 280) = 0;
  v12 = &v10[v11];
  while ( v12 != v10 )
  {
    v13 = *v10++;
    _libc_free(v13);
  }
  v14 = *(unsigned __int64 **)(a1 + 352);
  v15 = (unsigned __int64)&v14[2 * *(unsigned int *)(a1 + 360)];
  if ( v14 != (unsigned __int64 *)v15 )
  {
    do
    {
      v16 = *v14;
      v14 += 2;
      _libc_free(v16);
    }
    while ( v14 != (unsigned __int64 *)v15 );
    v15 = *(_QWORD *)(a1 + 352);
  }
  if ( v15 != a1 + 368 )
    _libc_free(v15);
  v17 = *(_QWORD *)(a1 + 304);
  if ( v17 != a1 + 320 )
    _libc_free(v17);
  j___libc_free_0(*(_QWORD *)(a1 + 248));
  v18 = *(unsigned __int64 **)(a1 + 152);
  v19 = *(unsigned int *)(a1 + 160);
  *(_QWORD *)(a1 + 128) = 0;
  v20 = &v18[v19];
  while ( v20 != v18 )
  {
    v21 = *v18++;
    _libc_free(v21);
  }
  v22 = *(unsigned __int64 **)(a1 + 200);
  v23 = (unsigned __int64)&v22[2 * *(unsigned int *)(a1 + 208)];
  if ( v22 != (unsigned __int64 *)v23 )
  {
    do
    {
      v24 = *v22;
      v22 += 2;
      _libc_free(v24);
    }
    while ( v22 != (unsigned __int64 *)v23 );
    v23 = *(_QWORD *)(a1 + 200);
  }
  if ( v23 != a1 + 216 )
    _libc_free(v23);
  v25 = *(_QWORD *)(a1 + 152);
  if ( v25 != a1 + 168 )
    _libc_free(v25);
  result = j___libc_free_0(*(_QWORD *)(a1 + 96));
  v27 = *(_QWORD *)(a1 + 80);
  if ( v27 )
  {
    sub_18FC7B0(*(_QWORD *)(v27 + 608));
    v28 = *(_QWORD *)(v27 + 512);
    if ( v28 != v27 + 528 )
      _libc_free(v28);
    v29 = *(_QWORD *)(v27 + 424);
    if ( v29 != *(_QWORD *)(v27 + 416) )
      _libc_free(v29);
    v30 = *(_QWORD **)(v27 + 8);
    v31 = &v30[3 * *(unsigned int *)(v27 + 16)];
    if ( v30 != v31 )
    {
      do
      {
        v32 = *(v31 - 1);
        v31 -= 3;
        if ( v32 != 0 && v32 != -8 && v32 != -16 )
          sub_1649B30(v31);
      }
      while ( v30 != v31 );
      v31 = *(_QWORD **)(v27 + 8);
    }
    if ( v31 != (_QWORD *)(v27 + 24) )
      _libc_free((unsigned __int64)v31);
    return j_j___libc_free_0(v27, 640);
  }
  return result;
}
