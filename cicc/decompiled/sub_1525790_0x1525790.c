// Function: sub_1525790
// Address: 0x1525790
//
__int64 __fastcall sub_1525790(__int64 *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 result; // rax
  __int64 v11; // r13
  __int64 v12; // r12
  __int64 v13; // rdi

  v2 = a1[64];
  if ( v2 )
    j_j___libc_free_0(v2, a1[66] - v2);
  j___libc_free_0(a1[60]);
  j___libc_free_0(a1[56]);
  v3 = a1[52];
  if ( v3 )
    j_j___libc_free_0(v3, a1[54] - v3);
  j___libc_free_0(a1[49]);
  v4 = a1[45];
  if ( v4 )
    j_j___libc_free_0(v4, a1[47] - v4);
  j___libc_free_0(a1[42]);
  if ( (a1[37] & 1) == 0 )
    j___libc_free_0(a1[38]);
  j___libc_free_0(a1[33]);
  v5 = a1[29];
  if ( v5 )
    j_j___libc_free_0(v5, a1[31] - v5);
  v6 = a1[26];
  if ( v6 )
    j_j___libc_free_0(v6, a1[28] - v6);
  v7 = a1[23];
  if ( v7 )
    j_j___libc_free_0(v7, a1[25] - v7);
  sub_1524150(a1[19]);
  v8 = a1[14];
  if ( v8 )
    j_j___libc_free_0(v8, a1[16] - v8);
  j___libc_free_0(a1[11]);
  v9 = a1[7];
  if ( v9 )
    j_j___libc_free_0(v9, a1[9] - v9);
  result = j___libc_free_0(a1[4]);
  v11 = a1[1];
  v12 = *a1;
  if ( v11 != *a1 )
  {
    do
    {
      v13 = *(_QWORD *)(v12 + 16);
      if ( v13 )
        result = j_j___libc_free_0(v13, *(_QWORD *)(v12 + 32) - v13);
      v12 += 40;
    }
    while ( v11 != v12 );
    v12 = *a1;
  }
  if ( v12 )
    return j_j___libc_free_0(v12, a1[2] - v12);
  return result;
}
