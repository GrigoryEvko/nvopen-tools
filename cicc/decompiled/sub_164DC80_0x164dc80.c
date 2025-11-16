// Function: sub_164DC80
// Address: 0x164dc80
//
__int64 __fastcall sub_164DC80(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  __int64 v10; // r13
  _QWORD *v11; // r12
  _QWORD *v12; // r13
  __int64 v13; // r14
  __int64 v14; // rdi
  unsigned __int64 v15; // rdi

  j___libc_free_0(*(_QWORD *)(a1 + 1648));
  j___libc_free_0(*(_QWORD *)(a1 + 1616));
  v2 = *(_QWORD *)(a1 + 1456);
  if ( v2 != a1 + 1472 )
    _libc_free(v2);
  v3 = *(_QWORD *)(a1 + 1176);
  if ( v3 != *(_QWORD *)(a1 + 1168) )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 1112);
  if ( v4 != a1 + 1128 )
    _libc_free(v4);
  v5 = *(_QWORD *)(a1 + 832);
  if ( v5 != *(_QWORD *)(a1 + 824) )
    _libc_free(v5);
  v6 = *(_QWORD *)(a1 + 792);
  if ( v6 )
    j_j___libc_free_0(v6, *(_QWORD *)(a1 + 808) - v6);
  j___libc_free_0(*(_QWORD *)(a1 + 768));
  j___libc_free_0(*(_QWORD *)(a1 + 736));
  v7 = *(_QWORD *)(a1 + 672);
  if ( v7 != *(_QWORD *)(a1 + 664) )
    _libc_free(v7);
  j___libc_free_0(*(_QWORD *)(a1 + 632));
  v8 = *(_QWORD *)(a1 + 344);
  if ( v8 != *(_QWORD *)(a1 + 336) )
    _libc_free(v8);
  v9 = *(_QWORD *)(a1 + 176);
  if ( v9 != *(_QWORD *)(a1 + 168) )
    _libc_free(v9);
  v10 = *(unsigned int *)(a1 + 128);
  if ( (_DWORD)v10 )
  {
    v11 = *(_QWORD **)(a1 + 112);
    v12 = &v11[2 * v10];
    do
    {
      if ( *v11 != -16 && *v11 != -8 )
      {
        v13 = v11[1];
        if ( v13 )
        {
          v14 = *(_QWORD *)(v13 + 24);
          if ( v14 )
            j_j___libc_free_0(v14, *(_QWORD *)(v13 + 40) - v14);
          j_j___libc_free_0(v13, 56);
        }
      }
      v11 += 2;
    }
    while ( v12 != v11 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 112));
  v15 = *(_QWORD *)(a1 + 80);
  if ( v15 != a1 + 96 )
    _libc_free(v15);
  return sub_154BA40((__int64 *)(a1 + 16));
}
