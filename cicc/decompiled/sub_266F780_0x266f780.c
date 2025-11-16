// Function: sub_266F780
// Address: 0x266f780
//
void __fastcall sub_266F780(unsigned __int64 a1)
{
  unsigned __int64 v2; // rdi

  *(_QWORD *)a1 = off_4A1FC98;
  v2 = *(_QWORD *)(a1 + 48);
  if ( v2 != a1 + 64 )
    _libc_free(v2);
  sub_C7D6A0(*(_QWORD *)(a1 + 24), 8LL * *(unsigned int *)(a1 + 40), 8);
  j_j___libc_free_0(a1);
}
