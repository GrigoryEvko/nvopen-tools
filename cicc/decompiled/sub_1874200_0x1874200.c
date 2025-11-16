// Function: sub_1874200
// Address: 0x1874200
//
__int64 __fastcall sub_1874200(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12
  _QWORD *v3; // rdi
  _QWORD *v4; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_1874200(v1[3]);
      v3 = (_QWORD *)v1[15];
      v1 = (_QWORD *)v1[2];
      sub_1874150(v3);
      v4 = (_QWORD *)v2[4];
      if ( v4 != v2 + 6 )
        j_j___libc_free_0(v4, v2[6] + 1LL);
      result = j_j___libc_free_0(v2, 152);
    }
    while ( v1 );
  }
  return result;
}
