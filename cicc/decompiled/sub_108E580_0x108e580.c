// Function: sub_108E580
// Address: 0x108e580
//
__int64 __fastcall sub_108E580(_QWORD *a1)
{
  __int64 v2; // rbx
  __int64 v3; // r14
  __int64 v4; // r12
  __int64 v5; // rdi

  v2 = a1[10];
  *a1 = off_497C0B0;
  if ( v2 )
  {
    v3 = (__int64)(a1 + 8);
    do
    {
      v4 = v2;
      sub_108E240(v3, *(_QWORD **)(v2 + 24));
      v5 = *(_QWORD *)(v2 + 64);
      v2 = *(_QWORD *)(v2 + 16);
      if ( v5 )
        j_j___libc_free_0(v5, *(_QWORD *)(v4 + 80) - v5);
      j_j___libc_free_0(v4, 88);
    }
    while ( v2 );
  }
  return j_j___libc_free_0(a1, 120);
}
