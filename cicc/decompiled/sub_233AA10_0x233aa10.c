// Function: sub_233AA10
// Address: 0x233aa10
//
void __fastcall sub_233AA10(unsigned __int64 *a1)
{
  volatile signed __int32 *v2; // rdi
  unsigned __int64 v3; // rdi

  v2 = (volatile signed __int32 *)a1[9];
  if ( v2 && !_InterlockedSub(v2 + 2, 1u) )
    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v2 + 8LL))(v2);
  v3 = a1[4];
  if ( (unsigned __int64 *)v3 != a1 + 6 )
    j_j___libc_free_0(v3);
  if ( (unsigned __int64 *)*a1 != a1 + 2 )
    j_j___libc_free_0(*a1);
}
