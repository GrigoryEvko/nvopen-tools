// Function: sub_1904240
// Address: 0x1904240
//
__int64 __fastcall sub_1904240(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 v3; // r12
  __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // rdi
  __int64 v9; // rdi

  *a1 = off_49F3300;
  v2 = a1[50];
  if ( v2 )
    j_j___libc_free_0(v2, a1[52] - v2);
  j___libc_free_0(a1[47]);
  v3 = a1[42];
  while ( v3 )
  {
    sub_1903F40(*(_QWORD *)(v3 + 24));
    v4 = v3;
    v3 = *(_QWORD *)(v3 + 16);
    j_j___libc_free_0(v4, 56);
  }
  v5 = a1[29];
  if ( v5 != a1[28] )
    _libc_free(v5);
  v6 = a1[25];
  v7 = a1[24];
  if ( v6 != v7 )
  {
    do
    {
      if ( *(_DWORD *)(v7 + 32) > 0x40u )
      {
        v8 = *(_QWORD *)(v7 + 24);
        if ( v8 )
          j_j___libc_free_0_0(v8);
      }
      if ( *(_DWORD *)(v7 + 16) > 0x40u )
      {
        v9 = *(_QWORD *)(v7 + 8);
        if ( v9 )
          j_j___libc_free_0_0(v9);
      }
      v7 += 40;
    }
    while ( v6 != v7 );
    v7 = a1[24];
  }
  if ( v7 )
    j_j___libc_free_0(v7, a1[26] - v7);
  j___libc_free_0(a1[21]);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 432);
}
