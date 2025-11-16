// Function: sub_12496B0
// Address: 0x12496b0
//
void __fastcall sub_12496B0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // rdi

  v3 = a1[1];
  v4 = *a1;
  if ( v3 != *a1 )
  {
    do
    {
      v5 = *(_QWORD *)(v4 + 8);
      if ( v5 != v4 + 32 )
        _libc_free(v5, a2);
      v4 += 72;
    }
    while ( v3 != v4 );
    v4 = *a1;
  }
  if ( v4 )
    j_j___libc_free_0(v4, a1[2] - v4);
}
