// Function: sub_255CE70
// Address: 0x255ce70
//
void __fastcall sub_255CE70(_QWORD *a1)
{
  unsigned __int64 v1; // r14
  unsigned __int64 v3; // rbx
  __int64 v4; // r13
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi

  v1 = (unsigned __int64)(a1 - 11);
  v3 = a1[5];
  *(a1 - 11) = &unk_4A16E18;
  *a1 = &unk_4A16DD8;
  if ( v3 )
  {
    v4 = (__int64)(a1 + 3);
    do
    {
      sub_255C230(v4, *(_QWORD *)(v3 + 24));
      v5 = v3;
      v3 = *(_QWORD *)(v3 + 16);
      j_j___libc_free_0(v5);
    }
    while ( v3 );
  }
  v6 = *(a1 - 6);
  *(a1 - 11) = &unk_4A16C00;
  if ( (_QWORD *)v6 != a1 - 4 )
    _libc_free(v6);
  sub_C7D6A0(*(a1 - 9), 8LL * *((unsigned int *)a1 - 14), 8);
  j_j___libc_free_0(v1);
}
