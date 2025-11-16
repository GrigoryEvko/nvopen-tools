// Function: sub_255CA20
// Address: 0x255ca20
//
void __fastcall sub_255CA20(unsigned __int64 a1)
{
  unsigned __int64 v2; // rbx
  __int64 v3; // r13
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  v2 = *(_QWORD *)(a1 + 128);
  *(_QWORD *)a1 = &unk_4A16E18;
  *(_QWORD *)(a1 + 88) = &unk_4A16DD8;
  if ( v2 )
  {
    v3 = a1 + 112;
    do
    {
      sub_255C230(v3, *(_QWORD *)(v2 + 24));
      v4 = v2;
      v2 = *(_QWORD *)(v2 + 16);
      j_j___libc_free_0(v4);
    }
    while ( v2 );
  }
  v5 = *(_QWORD *)(a1 + 40);
  *(_QWORD *)a1 = &unk_4A16C00;
  if ( v5 != a1 + 56 )
    _libc_free(v5);
  sub_C7D6A0(*(_QWORD *)(a1 + 16), 8LL * *(unsigned int *)(a1 + 32), 8);
  j_j___libc_free_0(a1);
}
