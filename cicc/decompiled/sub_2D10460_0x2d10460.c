// Function: sub_2D10460
// Address: 0x2d10460
//
void __fastcall sub_2D10460(__int64 a1)
{
  _QWORD *v2; // rbx
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  _QWORD *v5; // rbx
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // rdi

  v2 = *(_QWORD **)(a1 + 176);
  while ( v2 )
  {
    v3 = (unsigned __int64)v2;
    v2 = (_QWORD *)*v2;
    j_j___libc_free_0(v3);
  }
  memset(*(void **)(a1 + 160), 0, 8LL * *(_QWORD *)(a1 + 168));
  v4 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  if ( v4 != a1 + 208 )
    j_j___libc_free_0(v4);
  v5 = *(_QWORD **)(a1 + 120);
  while ( v5 )
  {
    v6 = (unsigned __int64)v5;
    v5 = (_QWORD *)*v5;
    j_j___libc_free_0(v6);
  }
  memset(*(void **)(a1 + 104), 0, 8LL * *(_QWORD *)(a1 + 112));
  v7 = *(_QWORD *)(a1 + 104);
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  if ( v7 != a1 + 152 )
    j_j___libc_free_0(v7);
  v8 = *(_QWORD *)(a1 + 72);
  while ( v8 )
  {
    sub_2D0FC20(*(_QWORD *)(v8 + 24));
    v9 = v8;
    v8 = *(_QWORD *)(v8 + 16);
    j_j___libc_free_0(v9);
  }
  v10 = *(_QWORD *)(a1 + 24);
  while ( v10 )
  {
    sub_2D0FC20(*(_QWORD *)(v10 + 24));
    v11 = v10;
    v10 = *(_QWORD *)(v10 + 16);
    j_j___libc_free_0(v11);
  }
}
