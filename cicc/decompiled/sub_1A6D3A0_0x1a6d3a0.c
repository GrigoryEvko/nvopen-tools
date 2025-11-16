// Function: sub_1A6D3A0
// Address: 0x1a6d3a0
//
void *__fastcall sub_1A6D3A0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // rax
  _QWORD *v4; // rbx
  _QWORD *v5; // r12
  unsigned __int64 v6; // rdi
  __int64 v7; // rax
  _QWORD *v8; // rbx
  _QWORD *v9; // r12
  __int64 v10; // rbx
  __int64 v11; // r12
  unsigned __int64 v12; // rdi
  __int64 v13; // rax
  _QWORD *v14; // r15
  _QWORD *v15; // r14
  __int64 v16; // rbx
  __int64 v17; // r12
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi

  *(_QWORD *)a1 = off_49F57C8;
  v2 = *(_QWORD *)(a1 + 680);
  if ( v2 != a1 + 696 )
    _libc_free(v2);
  v3 = *(unsigned int *)(a1 + 672);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD **)(a1 + 656);
    v5 = &v4[5 * v3];
    do
    {
      if ( *v4 != -16 && *v4 != -8 )
        j___libc_free_0(v4[2]);
      v4 += 5;
    }
    while ( v5 != v4 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 656));
  j___libc_free_0(*(_QWORD *)(a1 + 624));
  v6 = *(_QWORD *)(a1 + 536);
  if ( v6 != a1 + 552 )
    _libc_free(v6);
  v7 = *(unsigned int *)(a1 + 528);
  if ( (_DWORD)v7 )
  {
    v8 = *(_QWORD **)(a1 + 512);
    v9 = &v8[5 * v7];
    do
    {
      if ( *v8 != -8 && *v8 != -16 )
        j___libc_free_0(v8[2]);
      v8 += 5;
    }
    while ( v9 != v8 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 512));
  v10 = *(_QWORD *)(a1 + 488);
  v11 = *(_QWORD *)(a1 + 480);
  if ( v10 != v11 )
  {
    do
    {
      v12 = *(_QWORD *)(v11 + 8);
      if ( v12 != v11 + 24 )
        _libc_free(v12);
      v11 += 88;
    }
    while ( v10 != v11 );
    v11 = *(_QWORD *)(a1 + 480);
  }
  if ( v11 )
    j_j___libc_free_0(v11, *(_QWORD *)(a1 + 496) - v11);
  j___libc_free_0(*(_QWORD *)(a1 + 456));
  v13 = *(unsigned int *)(a1 + 440);
  if ( (_DWORD)v13 )
  {
    v14 = *(_QWORD **)(a1 + 424);
    v15 = &v14[8 * v13];
    do
    {
      if ( *v14 != -8 && *v14 != -16 )
      {
        v16 = v14[6];
        v17 = v14[5];
        if ( v16 != v17 )
        {
          do
          {
            v18 = *(_QWORD *)(v17 + 8);
            if ( v18 != v17 + 24 )
              _libc_free(v18);
            v17 += 56;
          }
          while ( v16 != v17 );
          v17 = v14[5];
        }
        if ( v17 )
          j_j___libc_free_0(v17, v14[7] - v17);
        j___libc_free_0(v14[2]);
      }
      v14 += 8;
    }
    while ( v15 != v14 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 424));
  v19 = *(_QWORD *)(a1 + 328);
  if ( v19 != *(_QWORD *)(a1 + 320) )
    _libc_free(v19);
  v20 = *(_QWORD *)(a1 + 232);
  if ( v20 != a1 + 248 )
    _libc_free(v20);
  *(_QWORD *)a1 = &unk_49EBE70;
  return sub_16366C0((_QWORD *)a1);
}
