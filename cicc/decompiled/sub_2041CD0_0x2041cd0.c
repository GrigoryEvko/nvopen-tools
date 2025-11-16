// Function: sub_2041CD0
// Address: 0x2041cd0
//
__int64 __fastcall sub_2041CD0(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 v3; // r13
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // rdi

  *a1 = &unk_49FFF60;
  v2 = a1[21];
  if ( v2 )
    j_j___libc_free_0(v2, a1[23] - v2);
  v3 = a1[20];
  if ( v3 )
  {
    j___libc_free_0(*(_QWORD *)(v3 + 40));
    j_j___libc_free_0(v3, 64);
  }
  v4 = a1[12];
  if ( v4 )
    j_j___libc_free_0(v4, a1[14] - v4);
  v5 = a1[9];
  if ( v5 )
    j_j___libc_free_0(v5, a1[11] - v5);
  v6 = a1[6];
  if ( v6 )
    j_j___libc_free_0(v6, a1[8] - v6);
  v7 = a1[3];
  if ( v7 )
    j_j___libc_free_0(v7, a1[5] - v7);
  return j_j___libc_free_0(a1, 200);
}
