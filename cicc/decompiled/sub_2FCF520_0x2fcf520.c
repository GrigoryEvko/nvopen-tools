// Function: sub_2FCF520
// Address: 0x2fcf520
//
void __fastcall sub_2FCF520(_QWORD *a1)
{
  unsigned __int64 v2; // r13
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi

  sub_2E0AFD0((__int64)(a1 + 2));
  v2 = a1[14];
  if ( v2 )
  {
    v3 = *(_QWORD *)(v2 + 16);
    while ( v3 )
    {
      sub_2FCF350(*(_QWORD *)(v3 + 24));
      v4 = v3;
      v3 = *(_QWORD *)(v3 + 16);
      j_j___libc_free_0(v4);
    }
    j_j___libc_free_0(v2);
  }
  v5 = a1[10];
  if ( (_QWORD *)v5 != a1 + 12 )
    _libc_free(v5);
  v6 = a1[2];
  if ( (_QWORD *)v6 != a1 + 4 )
    _libc_free(v6);
  j_j___libc_free_0((unsigned __int64)a1);
}
