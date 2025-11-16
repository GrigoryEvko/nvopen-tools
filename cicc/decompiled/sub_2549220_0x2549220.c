// Function: sub_2549220
// Address: 0x2549220
//
void __fastcall sub_2549220(_QWORD *a1)
{
  unsigned __int64 **v1; // r14
  unsigned __int64 **v3; // r12
  unsigned __int64 *v4; // rbx
  unsigned __int64 v5; // r15
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // [rsp+8h] [rbp-38h]

  v1 = (unsigned __int64 **)(a1 + 10);
  v3 = (unsigned __int64 **)(a1 + 2);
  v8 = (unsigned __int64)(a1 - 11);
  *(a1 - 11) = &unk_4A1BC90;
  *a1 = &unk_4A1BD28;
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
  v7 = *(a1 - 6);
  *(a1 - 11) = &unk_4A16C00;
  if ( (_QWORD *)v7 != a1 - 4 )
    _libc_free(v7);
  sub_C7D6A0(*(a1 - 9), 8LL * *((unsigned int *)a1 - 14), 8);
  j_j___libc_free_0(v8);
}
