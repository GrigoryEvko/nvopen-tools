// Function: sub_255C340
// Address: 0x255c340
//
__int64 __fastcall sub_255C340(_QWORD *a1)
{
  unsigned __int64 v2; // rbx
  __int64 v3; // r13
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  v2 = a1[5];
  *(a1 - 11) = &unk_4A16E18;
  *a1 = &unk_4A16DD8;
  if ( v2 )
  {
    v3 = (__int64)(a1 + 3);
    do
    {
      sub_255C230(v3, *(_QWORD *)(v2 + 24));
      v4 = v2;
      v2 = *(_QWORD *)(v2 + 16);
      j_j___libc_free_0(v4);
    }
    while ( v2 );
  }
  v5 = *(a1 - 6);
  *(a1 - 11) = &unk_4A16C00;
  if ( (_QWORD *)v5 != a1 - 4 )
    _libc_free(v5);
  return sub_C7D6A0(*(a1 - 9), 8LL * *((unsigned int *)a1 - 14), 8);
}
