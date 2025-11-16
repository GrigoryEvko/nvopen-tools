// Function: sub_393DAE0
// Address: 0x393dae0
//
void __fastcall sub_393DAE0(unsigned __int64 a1)
{
  unsigned __int64 v2; // rdi

  v2 = *(_QWORD *)(a1 + 8);
  if ( v2 )
    j_j___libc_free_0(v2);
  j_j___libc_free_0(a1);
}
