// Function: sub_2E39780
// Address: 0x2e39780
//
void __fastcall sub_2E39780(__int64 a1)
{
  __int64 v2; // rsi
  unsigned __int64 v3; // rdi
  _QWORD *v4; // rbx
  _QWORD *v5; // r12
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  _QWORD *v10; // r12
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi

  v2 = *(unsigned int *)(a1 + 184);
  *(_QWORD *)a1 = &unk_4A289B8;
  sub_C7D6A0(*(_QWORD *)(a1 + 168), 16 * v2, 8);
  v3 = *(_QWORD *)(a1 + 136);
  if ( v3 )
    j_j___libc_free_0(v3);
  v4 = *(_QWORD **)(a1 + 88);
  *(_QWORD *)a1 = &unk_49E5580;
  while ( (_QWORD *)(a1 + 88) != v4 )
  {
    v5 = v4;
    v4 = (_QWORD *)*v4;
    v6 = v5[18];
    if ( (_QWORD *)v6 != v5 + 20 )
      _libc_free(v6);
    v7 = v5[14];
    if ( (_QWORD *)v7 != v5 + 16 )
      _libc_free(v7);
    v8 = v5[4];
    if ( (_QWORD *)v8 != v5 + 6 )
      _libc_free(v8);
    j_j___libc_free_0((unsigned __int64)v5);
  }
  v9 = *(_QWORD *)(a1 + 64);
  if ( v9 )
    j_j___libc_free_0(v9);
  v10 = *(_QWORD **)(a1 + 32);
  while ( (_QWORD *)(a1 + 32) != v10 )
  {
    v11 = (unsigned __int64)v10;
    v10 = (_QWORD *)*v10;
    j_j___libc_free_0(v11);
  }
  v12 = *(_QWORD *)(a1 + 8);
  if ( v12 )
    j_j___libc_free_0(v12);
}
