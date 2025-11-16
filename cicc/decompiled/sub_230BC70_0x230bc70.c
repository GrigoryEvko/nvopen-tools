// Function: sub_230BC70
// Address: 0x230bc70
//
void __fastcall sub_230BC70(__int64 a1)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi

  *(_QWORD *)a1 = &unk_4A0B470;
  sub_230BB90(*(_QWORD **)(a1 + 120));
  v2 = *(_QWORD *)(a1 + 72);
  while ( v2 )
  {
    v3 = v2;
    sub_23092F0(*(_QWORD **)(v2 + 24));
    v4 = *(_QWORD *)(v2 + 40);
    v2 = *(_QWORD *)(v2 + 16);
    if ( v4 != v3 + 56 )
      _libc_free(v4);
    j_j___libc_free_0(v3);
  }
  sub_230BAD0(*(_QWORD **)(a1 + 24));
}
