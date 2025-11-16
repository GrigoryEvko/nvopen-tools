// Function: sub_2549750
// Address: 0x2549750
//
void __fastcall sub_2549750(unsigned __int64 a1)
{
  unsigned __int64 **v1; // r15
  unsigned __int64 **v3; // r13
  unsigned __int64 *v4; // r12
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi

  v1 = (unsigned __int64 **)(a1 + 168);
  v3 = (unsigned __int64 **)(a1 + 104);
  *(_QWORD *)a1 = &unk_4A1BC90;
  *(_QWORD *)(a1 + 88) = &unk_4A1BD28;
  do
  {
    v4 = *v3;
    if ( *v3 )
    {
      v5 = v4[12];
      while ( v5 )
      {
        sub_253AC60(*(_QWORD *)(v5 + 24));
        v6 = v5;
        v5 = *(_QWORD *)(v5 + 16);
        j_j___libc_free_0(v6);
      }
      if ( (unsigned __int64 *)*v4 != v4 + 2 )
        _libc_free(*v4);
    }
    ++v3;
  }
  while ( v1 != v3 );
  v7 = *(_QWORD *)(a1 + 40);
  *(_QWORD *)a1 = &unk_4A16C00;
  if ( v7 != a1 + 56 )
    _libc_free(v7);
  sub_C7D6A0(*(_QWORD *)(a1 + 16), 8LL * *(unsigned int *)(a1 + 32), 8);
  j_j___libc_free_0(a1);
}
