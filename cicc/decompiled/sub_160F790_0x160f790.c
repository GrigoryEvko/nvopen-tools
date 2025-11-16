// Function: sub_160F790
// Address: 0x160f790
//
__int64 __fastcall sub_160F790(_QWORD *a1)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  __int64 v4; // rdi

  v2 = a1[75];
  v3 = a1[76];
  *a1 = off_49EDD08;
  a1[20] = &unk_49EDDC8;
  if ( v2 != v3 )
  {
    do
    {
      v4 = *(_QWORD *)(v2 + 8);
      if ( v4 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 8LL))(v4);
      v2 += 16;
    }
    while ( v3 != v2 );
    v3 = a1[75];
  }
  if ( v3 )
    j_j___libc_free_0(v3, a1[77] - v3);
  j___libc_free_0(a1[72]);
  sub_160F3F0((__int64)(a1 + 20));
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 640);
}
