// Function: sub_36D78B0
// Address: 0x36d78b0
//
void __fastcall sub_36D78B0(unsigned __int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // rdi

  *(_QWORD *)a1 = off_4A3BE98;
  v7 = *(_QWORD *)(a1 + 1056);
  if ( v7 != a1 + 1072 )
    _libc_free(v7);
  if ( (*(_BYTE *)(a1 + 984) & 1) == 0 )
  {
    a2 = 8LL * *(unsigned int *)(a1 + 1000);
    sub_C7D6A0(*(_QWORD *)(a1 + 992), a2, 4);
  }
  sub_3424860(a1, a2, a3, a4, a5, a6);
  j_j___libc_free_0(a1);
}
