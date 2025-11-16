// Function: sub_1857740
// Address: 0x1857740
//
__int64 __fastcall sub_1857740(__int64 a1)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rax
  _QWORD *v7; // rbx
  _QWORD *v8; // r13
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi

  v2 = *(_QWORD **)(a1 + 560);
  *(_QWORD *)a1 = off_49F13C8;
  while ( v2 )
  {
    v3 = v2;
    v2 = (_QWORD *)*v2;
    j_j___libc_free_0(v3, 24);
  }
  memset(*(void **)(a1 + 544), 0, 8LL * *(_QWORD *)(a1 + 552));
  v4 = *(_QWORD *)(a1 + 544);
  *(_QWORD *)(a1 + 568) = 0;
  *(_QWORD *)(a1 + 560) = 0;
  if ( v4 != a1 + 592 )
    j_j___libc_free_0(v4, 8LL * *(_QWORD *)(a1 + 552));
  sub_1857560(a1 + 488);
  v5 = *(_QWORD *)(a1 + 488);
  if ( v5 != a1 + 536 )
    j_j___libc_free_0(v5, 8LL * *(_QWORD *)(a1 + 496));
  v6 = *(unsigned int *)(a1 + 480);
  if ( (_DWORD)v6 )
  {
    v7 = *(_QWORD **)(a1 + 464);
    v8 = &v7[10 * v6];
    do
    {
      if ( *v7 != -16 && *v7 != -8 )
      {
        v9 = v7[3];
        if ( v9 != v7[2] )
          _libc_free(v9);
      }
      v7 += 10;
    }
    while ( v8 != v7 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 464));
  v10 = *(_QWORD *)(a1 + 176);
  if ( v10 != *(_QWORD *)(a1 + 168) )
    _libc_free(v10);
  sub_1636790((_QWORD *)a1);
  return j_j___libc_free_0(a1, 600);
}
