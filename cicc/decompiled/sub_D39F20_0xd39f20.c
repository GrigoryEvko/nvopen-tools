// Function: sub_D39F20
// Address: 0xd39f20
//
__int64 __fastcall sub_D39F20(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rbx
  unsigned __int64 v3; // r14
  _QWORD *v4; // r12
  __int64 result; // rax
  _QWORD *v6; // r12
  unsigned __int64 v7; // rbx
  __int64 v8; // rdi

  v2 = *(_QWORD **)a1;
  v3 = *(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 6);
  if ( *(_QWORD *)a1 != v3 )
  {
    v4 = (_QWORD *)a2;
    do
    {
      if ( v4 )
      {
        a2 = (__int64)(v2 + 1);
        *v4 = *v2;
        result = sub_D38730((__int64)(v4 + 1), (__int64)(v2 + 1));
      }
      v2 += 8;
      v4 += 8;
    }
    while ( (_QWORD *)v3 != v2 );
    v6 = *(_QWORD **)a1;
    v7 = *(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 6);
    if ( *(_QWORD *)a1 != v7 )
    {
      do
      {
        v7 -= 64LL;
        v8 = *(_QWORD *)(v7 + 40);
        if ( v8 != v7 + 56 )
          _libc_free(v8, a2);
        a2 = 8LL * *(unsigned int *)(v7 + 32);
        result = sub_C7D6A0(*(_QWORD *)(v7 + 16), a2, 8);
      }
      while ( (_QWORD *)v7 != v6 );
    }
  }
  return result;
}
