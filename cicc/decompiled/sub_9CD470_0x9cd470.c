// Function: sub_9CD470
// Address: 0x9cd470
//
__int64 __fastcall sub_9CD470(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // r13
  _QWORD *v3; // r14
  _QWORD *v4; // rbx
  __int64 *v5; // r12
  __int64 *v6; // r15
  __int64 v7; // rdi
  __int64 v8; // r12
  __int64 v9; // r15
  __int64 v10; // rdi
  __int64 result; // rax

  v2 = *(_QWORD **)(a1 + 8);
  if ( v2 != a2 )
  {
    v3 = a2;
    v4 = a2;
    do
    {
      v5 = (__int64 *)v4[12];
      v6 = (__int64 *)v4[11];
      if ( v5 != v6 )
      {
        do
        {
          v7 = *v6;
          if ( *v6 )
          {
            a2 = (_QWORD *)(v6[2] - v7);
            j_j___libc_free_0(v7, a2);
          }
          v6 += 3;
        }
        while ( v5 != v6 );
        v6 = (__int64 *)v4[11];
      }
      if ( v6 )
      {
        a2 = (_QWORD *)(v4[13] - (_QWORD)v6);
        j_j___libc_free_0(v6, a2);
      }
      v8 = v4[9];
      v9 = v4[8];
      if ( v8 != v9 )
      {
        do
        {
          v10 = *(_QWORD *)(v9 + 8);
          if ( v10 != v9 + 24 )
            _libc_free(v10, a2);
          v9 += 72;
        }
        while ( v8 != v9 );
        v9 = v4[8];
      }
      if ( v9 )
      {
        a2 = (_QWORD *)(v4[10] - v9);
        j_j___libc_free_0(v9, a2);
      }
      if ( (_QWORD *)*v4 != v4 + 3 )
        _libc_free(*v4, a2);
      v4 += 14;
    }
    while ( v2 != v4 );
    *(_QWORD *)(a1 + 8) = v3;
    return a1;
  }
  return result;
}
