// Function: sub_1874150
// Address: 0x1874150
//
__int64 __fastcall sub_1874150(_QWORD *a1)
{
  _QWORD *v1; // r13
  _QWORD *v2; // r14
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // rdi
  _QWORD *v6; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_1874150(v1[3]);
      v3 = v1[12];
      v1 = (_QWORD *)v1[2];
      while ( v3 )
      {
        v4 = v3;
        sub_1873E80(*(_QWORD **)(v3 + 24));
        v5 = *(_QWORD *)(v3 + 32);
        v3 = *(_QWORD *)(v3 + 16);
        if ( v5 )
          j_j___libc_free_0(v5, *(_QWORD *)(v4 + 48) - v5);
        j_j___libc_free_0(v4, 80);
      }
      v6 = (_QWORD *)v2[6];
      if ( v6 != v2 + 8 )
        j_j___libc_free_0(v6, v2[8] + 1LL);
      result = j_j___libc_free_0(v2, 128);
    }
    while ( v1 );
  }
  return result;
}
