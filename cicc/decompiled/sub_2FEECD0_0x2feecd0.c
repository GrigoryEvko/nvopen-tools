// Function: sub_2FEECD0
// Address: 0x2feecd0
//
void __fastcall sub_2FEECD0(__int64 a1)
{
  unsigned __int64 v1; // r12
  __int64 v2; // rdi

  v1 = a1 - 8;
  v2 = a1 + 32;
  qword_5023870 = 0;
  if ( *(_QWORD *)(v2 - 16) != v2 )
    _libc_free(*(_QWORD *)(v2 - 16));
  j_j___libc_free_0(v1);
}
