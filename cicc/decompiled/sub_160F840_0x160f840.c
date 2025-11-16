// Function: sub_160F840
// Address: 0x160f840
//
__int64 __fastcall sub_160F840(_QWORD *a1)
{
  _QWORD *v1; // r14
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // rdi

  v1 = a1 - 20;
  v3 = a1[55];
  v4 = a1[56];
  *(a1 - 20) = off_49EDD08;
  *a1 = &unk_49EDDC8;
  if ( v3 != v4 )
  {
    do
    {
      v5 = *(_QWORD *)(v3 + 8);
      if ( v5 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v5 + 8LL))(v5);
      v3 += 16;
    }
    while ( v4 != v3 );
    v4 = a1[55];
  }
  if ( v4 )
    j_j___libc_free_0(v4, a1[57] - v4);
  j___libc_free_0(a1[52]);
  sub_160F3F0((__int64)a1);
  sub_16366C0(v1);
  return j_j___libc_free_0(v1, 640);
}
