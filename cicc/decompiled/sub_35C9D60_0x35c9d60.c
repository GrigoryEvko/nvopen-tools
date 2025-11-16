// Function: sub_35C9D60
// Address: 0x35c9d60
//
void __fastcall sub_35C9D60(unsigned __int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  __int64 v9; // r13
  __int64 i; // rbx
  unsigned __int64 v11; // rdi

  *(_QWORD *)a1 = off_4A3A7F8;
  v2 = *(_QWORD *)(a1 + 728);
  if ( v2 != a1 + 744 )
    _libc_free(v2);
  v3 = *(_QWORD *)(a1 + 640);
  if ( v3 != a1 + 656 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 616), 4LL * *(unsigned int *)(a1 + 632), 4);
  v4 = *(_QWORD *)(a1 + 496);
  if ( v4 )
    j_j___libc_free_0_0(v4);
  v5 = *(_QWORD *)(a1 + 424);
  if ( v5 != a1 + 440 )
    _libc_free(v5);
  v6 = *(_QWORD *)(a1 + 352);
  if ( v6 != a1 + 368 )
    _libc_free(v6);
  v7 = *(_QWORD *)(a1 + 288);
  if ( v7 != a1 + 312 )
    _libc_free(v7);
  v8 = *(_QWORD *)(a1 + 232);
  if ( v8 != a1 + 256 )
    _libc_free(v8);
  v9 = *(_QWORD *)(a1 + 200);
  if ( v9 )
  {
    for ( i = v9 + 24LL * *(_QWORD *)(v9 - 8); v9 != i; i -= 24 )
    {
      v11 = *(_QWORD *)(i - 8);
      if ( v11 )
        j_j___libc_free_0_0(v11);
    }
    j_j_j___libc_free_0_0(v9 - 8);
  }
  *(_QWORD *)a1 = &unk_49DAF80;
  sub_BB9100(a1);
  j_j___libc_free_0(a1);
}
