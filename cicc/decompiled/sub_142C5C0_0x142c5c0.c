// Function: sub_142C5C0
// Address: 0x142c5c0
//
void __fastcall sub_142C5C0(_QWORD *a1)
{
  _QWORD *v1; // r13
  __int64 v3; // r14
  __int64 v4; // r12
  __int64 v5; // rdi
  __int64 v6; // r14
  __int64 v7; // r12
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // rdi

  v1 = (_QWORD *)a1[12];
  *a1 = &unk_49EB4B8;
  if ( v1 )
  {
    v3 = v1[13];
    v4 = v1[12];
    if ( v3 != v4 )
    {
      do
      {
        v5 = *(_QWORD *)(v4 + 16);
        if ( v5 )
          j_j___libc_free_0(v5, *(_QWORD *)(v4 + 32) - v5);
        v4 += 40;
      }
      while ( v3 != v4 );
      v4 = v1[12];
    }
    if ( v4 )
      j_j___libc_free_0(v4, v1[14] - v4);
    v6 = v1[10];
    v7 = v1[9];
    if ( v6 != v7 )
    {
      do
      {
        v8 = *(_QWORD *)(v7 + 16);
        if ( v8 )
          j_j___libc_free_0(v8, *(_QWORD *)(v7 + 32) - v8);
        v7 += 40;
      }
      while ( v6 != v7 );
      v7 = v1[9];
    }
    if ( v7 )
      j_j___libc_free_0(v7, v1[11] - v7);
    v9 = v1[6];
    if ( v9 )
      j_j___libc_free_0(v9, v1[8] - v9);
    v10 = v1[3];
    if ( v10 )
      j_j___libc_free_0(v10, v1[5] - v10);
    if ( *v1 )
      j_j___libc_free_0(*v1, v1[2] - *v1);
    j_j___libc_free_0(v1, 120);
  }
  v11 = a1[9];
  if ( v11 )
    j_j___libc_free_0(v11, a1[11] - v11);
  v12 = a1[5];
  *a1 = &unk_49EB478;
  if ( v12 )
    j_j___libc_free_0(v12, a1[7] - v12);
}
