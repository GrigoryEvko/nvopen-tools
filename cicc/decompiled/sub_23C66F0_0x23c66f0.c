// Function: sub_23C66F0
// Address: 0x23c66f0
//
void __fastcall sub_23C66F0(unsigned __int64 *a1)
{
  volatile signed __int32 *v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  v2 = (volatile signed __int32 *)a1[18];
  if ( v2 && !_InterlockedSub(v2 + 2, 1u) )
    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v2 + 8LL))(v2);
  v3 = a1[12];
  if ( (unsigned __int64 *)v3 != a1 + 14 )
    j_j___libc_free_0(v3);
  v4 = a1[8];
  if ( (unsigned __int64 *)v4 != a1 + 10 )
    j_j___libc_free_0(v4);
  v5 = a1[4];
  if ( (unsigned __int64 *)v5 != a1 + 6 )
    j_j___libc_free_0(v5);
  if ( (unsigned __int64 *)*a1 != a1 + 2 )
    j_j___libc_free_0(*a1);
}
