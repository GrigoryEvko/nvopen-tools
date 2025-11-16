// Function: sub_1DCAB80
// Address: 0x1dcab80
//
void *__fastcall sub_1DCAB80(__int64 a1)
{
  unsigned __int64 *v2; // rbx
  unsigned __int64 *v3; // r12
  __int64 v4; // rdi
  __int64 v5; // rdi
  _QWORD *v6; // rbx
  _QWORD *v7; // rdi
  __int64 v8; // rdi
  _QWORD *v9; // rbx
  _QWORD *v10; // rdi
  _QWORD *v11; // r15
  unsigned __int64 v12; // rdi
  _QWORD *v13; // r12
  __int64 v14; // rdi
  _QWORD *v15; // r14
  _QWORD *v16; // rbx
  _QWORD *v17; // rdi

  *(_QWORD *)a1 = &unk_49FAF68;
  j___libc_free_0(*(_QWORD *)(a1 + 448));
  v2 = *(unsigned __int64 **)(a1 + 424);
  v3 = *(unsigned __int64 **)(a1 + 416);
  if ( v2 != v3 )
  {
    do
    {
      if ( (unsigned __int64 *)*v3 != v3 + 2 )
        _libc_free(*v3);
      v3 += 4;
    }
    while ( v2 != v3 );
    v3 = *(unsigned __int64 **)(a1 + 416);
  }
  if ( v3 )
    j_j___libc_free_0(v3, *(_QWORD *)(a1 + 432) - (_QWORD)v3);
  v4 = *(_QWORD *)(a1 + 392);
  if ( v4 )
    j_j___libc_free_0(v4, *(_QWORD *)(a1 + 408) - v4);
  v5 = *(_QWORD *)(a1 + 368);
  if ( v5 )
    j_j___libc_free_0(v5, *(_QWORD *)(a1 + 384) - v5);
  v6 = *(_QWORD **)(a1 + 320);
  while ( v6 != (_QWORD *)(a1 + 320) )
  {
    v7 = v6;
    v6 = (_QWORD *)*v6;
    j_j___libc_free_0(v7, 40);
  }
  v8 = *(_QWORD *)(a1 + 280);
  if ( v8 )
    j_j___libc_free_0(v8, *(_QWORD *)(a1 + 296) - v8);
  v9 = *(_QWORD **)(a1 + 256);
  while ( v9 != (_QWORD *)(a1 + 256) )
  {
    v10 = v9;
    v9 = (_QWORD *)*v9;
    j_j___libc_free_0(v10, 40);
  }
  v11 = *(_QWORD **)(a1 + 232);
  v12 = (unsigned __int64)&v11[7 * *(unsigned int *)(a1 + 240)];
  v13 = (_QWORD *)(v12 - 48);
  if ( v11 != (_QWORD *)v12 )
  {
    do
    {
      v14 = v13[3];
      v15 = v13 - 1;
      if ( v14 )
        j_j___libc_free_0(v14, v13[5] - v14);
      v16 = (_QWORD *)*v13;
      while ( v16 != v13 )
      {
        v17 = v16;
        v16 = (_QWORD *)*v16;
        j_j___libc_free_0(v17, 40);
      }
      v13 -= 7;
    }
    while ( v11 != v15 );
    v12 = *(_QWORD *)(a1 + 232);
  }
  if ( v12 != a1 + 248 )
    _libc_free(v12);
  _libc_free(*(_QWORD *)(a1 + 208));
  _libc_free(*(_QWORD *)(a1 + 184));
  _libc_free(*(_QWORD *)(a1 + 160));
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0((_QWORD *)a1);
}
