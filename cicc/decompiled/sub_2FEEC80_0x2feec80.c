// Function: sub_2FEEC80
// Address: 0x2feec80
//
void __fastcall sub_2FEEC80(unsigned __int64 a1)
{
  unsigned __int64 v2; // rdi

  v2 = *(_QWORD *)(a1 + 24);
  qword_5023870 = 0;
  if ( v2 != a1 + 40 )
    _libc_free(v2);
  j_j___libc_free_0(a1);
}
