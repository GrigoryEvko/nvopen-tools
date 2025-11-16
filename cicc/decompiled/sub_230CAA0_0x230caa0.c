// Function: sub_230CAA0
// Address: 0x230caa0
//
void __fastcall sub_230CAA0(_QWORD *a1)
{
  volatile signed __int32 *v2; // rdi
  unsigned __int64 v3; // rdi

  *a1 = &unk_4A0E938;
  v2 = (volatile signed __int32 *)a1[5];
  if ( v2 && !_InterlockedSub(v2 + 2, 1u) )
    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v2 + 8LL))(v2);
  v3 = a1[1];
  if ( (_QWORD *)v3 != a1 + 3 )
    j_j___libc_free_0(v3);
  j_j___libc_free_0((unsigned __int64)a1);
}
