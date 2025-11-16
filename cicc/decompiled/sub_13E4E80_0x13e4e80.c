// Function: sub_13E4E80
// Address: 0x13e4e80
//
__int64 __fastcall sub_13E4E80(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rbx
  __int64 v4; // rdi

  *a1 = &unk_49EA6F8;
  v2 = a1[27];
  if ( v2 )
    j_j___libc_free_0(v2, a1[29] - v2);
  v3 = a1[22];
  while ( v3 )
  {
    sub_13E4CB0(*(_QWORD *)(v3 + 24));
    v4 = v3;
    v3 = *(_QWORD *)(v3 + 16);
    j_j___libc_free_0(v4, 48);
  }
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
