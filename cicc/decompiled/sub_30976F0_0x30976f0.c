// Function: sub_30976F0
// Address: 0x30976f0
//
void __fastcall sub_30976F0(unsigned __int64 a1)
{
  _QWORD *v2; // rdi
  _QWORD *v3; // rbx
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  __int64 v7; // rdi
  unsigned __int64 v8; // rdi

  v2 = (_QWORD *)(a1 + 184);
  *(v2 - 23) = &unk_4A31DB8;
  *v2 = &unk_4A0AA50;
  sub_22F3580((__int64)v2);
  v3 = *(_QWORD **)(a1 + 512);
  while ( v3 != (_QWORD *)(a1 + 512) )
  {
    v4 = (unsigned __int64)v3;
    v3 = (_QWORD *)*v3;
    v5 = *(_QWORD *)(v4 + 16);
    if ( v5 != v4 + 32 )
      j_j___libc_free_0(v5);
    j_j___libc_free_0(v4);
  }
  v6 = *(_QWORD *)(a1 + 368);
  if ( v6 != a1 + 384 )
    _libc_free(v6);
  v7 = *(_QWORD *)(a1 + 344);
  *(_QWORD *)(a1 + 184) = &unk_4A08310;
  sub_C7D6A0(v7, 12LL * *(unsigned int *)(a1 + 360), 4);
  v8 = *(_QWORD *)(a1 + 192);
  if ( v8 != a1 + 208 )
    _libc_free(v8);
  *(_QWORD *)a1 = &unk_4A082F0;
  sub_22F4E50(a1);
  j_j___libc_free_0(a1);
}
