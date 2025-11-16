// Function: sub_353EAD0
// Address: 0x353ead0
//
void __fastcall sub_353EAD0(_QWORD *a1)
{
  __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  __int64 v9; // r13
  __int64 i; // rbx
  unsigned __int64 v11; // rdi

  *a1 = &unk_4A39190;
  v2 = a1[98];
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  v3 = a1[74];
  if ( (_QWORD *)v3 != a1 + 76 )
    _libc_free(v3);
  v4 = a1[68];
  if ( v4 )
    j_j___libc_free_0_0(v4);
  v5 = a1[59];
  if ( (_QWORD *)v5 != a1 + 61 )
    _libc_free(v5);
  v6 = a1[50];
  if ( (_QWORD *)v6 != a1 + 52 )
    _libc_free(v6);
  v7 = a1[42];
  if ( (_QWORD *)v7 != a1 + 45 )
    _libc_free(v7);
  v8 = a1[35];
  if ( (_QWORD *)v8 != a1 + 38 )
    _libc_free(v8);
  v9 = a1[31];
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
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
