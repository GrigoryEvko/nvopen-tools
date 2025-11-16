// Function: sub_E62D10
// Address: 0xe62d10
//
__int64 __fastcall sub_E62D10(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r13
  __int64 v3; // r12
  __int64 v4; // rdi
  void *v5; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      v1 = (_QWORD *)*v1;
      v3 = v2[3];
      if ( v3 )
      {
        v4 = *(_QWORD *)(v3 + 56);
        if ( v4 )
          j_j___libc_free_0(v4, *(_QWORD *)(v3 + 72) - v4);
        sub_E62D10(*(_QWORD *)(v3 + 16));
        memset(*(void **)v3, 0, 8LL * *(_QWORD *)(v3 + 8));
        v5 = *(void **)v3;
        *(_QWORD *)(v3 + 24) = 0;
        *(_QWORD *)(v3 + 16) = 0;
        if ( v5 != (void *)(v3 + 48) )
          j_j___libc_free_0(v5, 8LL * *(_QWORD *)(v3 + 8));
        j_j___libc_free_0(v3, 96);
      }
      result = j_j___libc_free_0(v2, 40);
    }
    while ( v1 );
  }
  return result;
}
