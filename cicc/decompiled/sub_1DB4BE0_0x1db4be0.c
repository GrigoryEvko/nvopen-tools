// Function: sub_1DB4BE0
// Address: 0x1db4be0
//
void __fastcall sub_1DB4BE0(__int64 a1, unsigned __int64 *a2)
{
  unsigned __int64 v2; // r13
  __int64 v3; // rbx
  __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  v2 = a2[12];
  if ( v2 )
  {
    v3 = *(_QWORD *)(v2 + 16);
    while ( v3 )
    {
      sub_1DB3580(*(_QWORD *)(v3 + 24));
      v4 = v3;
      v3 = *(_QWORD *)(v3 + 16);
      j_j___libc_free_0(v4, 56);
    }
    j_j___libc_free_0(v2, 48);
  }
  v5 = a2[8];
  if ( (unsigned __int64 *)v5 != a2 + 10 )
    _libc_free(v5);
  if ( (unsigned __int64 *)*a2 != a2 + 2 )
    _libc_free(*a2);
}
