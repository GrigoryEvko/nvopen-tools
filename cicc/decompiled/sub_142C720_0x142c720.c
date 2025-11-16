// Function: sub_142C720
// Address: 0x142c720
//
__int64 __fastcall sub_142C720(_QWORD *a1)
{
  _QWORD *v2; // r14
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // rdi
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // rdi

  v2 = (_QWORD *)a1[12];
  *a1 = &unk_49EB4B8;
  if ( v2 )
  {
    v3 = v2[13];
    v4 = v2[12];
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
      v4 = v2[12];
    }
    if ( v4 )
      j_j___libc_free_0(v4, v2[14] - v4);
    v6 = v2[10];
    v7 = v2[9];
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
      v7 = v2[9];
    }
    if ( v7 )
      j_j___libc_free_0(v7, v2[11] - v7);
    v9 = v2[6];
    if ( v9 )
      j_j___libc_free_0(v9, v2[8] - v9);
    v10 = v2[3];
    if ( v10 )
      j_j___libc_free_0(v10, v2[5] - v10);
    if ( *v2 )
      j_j___libc_free_0(*v2, v2[2] - *v2);
    j_j___libc_free_0(v2, 120);
  }
  v11 = a1[9];
  if ( v11 )
    j_j___libc_free_0(v11, a1[11] - v11);
  v12 = a1[5];
  *a1 = &unk_49EB478;
  if ( v12 )
    j_j___libc_free_0(v12, a1[7] - v12);
  return j_j___libc_free_0(a1, 104);
}
