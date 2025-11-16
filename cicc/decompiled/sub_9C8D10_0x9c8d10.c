// Function: sub_9C8D10
// Address: 0x9c8d10
//
__int64 __fastcall sub_9C8D10(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  __int64 v4; // rdx
  __int64 v5; // r12
  __int64 v6; // r15
  __int64 v7; // rbx
  _QWORD *v8; // r14
  _QWORD *v9; // rbx
  _QWORD *v10; // r12
  _QWORD *v11; // rbx
  __int64 v12; // [rsp+8h] [rbp-38h]

  result = *(unsigned int *)(a1 + 8);
  v12 = *(_QWORD *)a1;
  v3 = *(_QWORD *)a1 + 192 * result;
  if ( *(_QWORD *)a1 != v3 )
  {
    do
    {
      v4 = *(unsigned int *)(v3 - 120);
      v5 = *(_QWORD *)(v3 - 128);
      v3 -= 192;
      v6 = v5 + 56 * v4;
      if ( v5 != v6 )
      {
        do
        {
          v7 = *(unsigned int *)(v6 - 40);
          v8 = *(_QWORD **)(v6 - 48);
          v6 -= 56;
          v9 = &v8[4 * v7];
          if ( v8 != v9 )
          {
            do
            {
              v9 -= 4;
              if ( (_QWORD *)*v9 != v9 + 2 )
              {
                a2 = v9[2] + 1LL;
                j_j___libc_free_0(*v9, a2);
              }
            }
            while ( v8 != v9 );
            v8 = *(_QWORD **)(v6 + 8);
          }
          if ( v8 != (_QWORD *)(v6 + 24) )
            _libc_free(v8, a2);
        }
        while ( v5 != v6 );
        v5 = *(_QWORD *)(v3 + 64);
      }
      if ( v5 != v3 + 80 )
        _libc_free(v5, a2);
      v10 = *(_QWORD **)(v3 + 16);
      v11 = &v10[4 * *(unsigned int *)(v3 + 24)];
      if ( v10 != v11 )
      {
        do
        {
          v11 -= 4;
          if ( (_QWORD *)*v11 != v11 + 2 )
          {
            a2 = v11[2] + 1LL;
            j_j___libc_free_0(*v11, a2);
          }
        }
        while ( v10 != v11 );
        v10 = *(_QWORD **)(v3 + 16);
      }
      if ( v10 != (_QWORD *)(v3 + 32) )
        _libc_free(v10, a2);
    }
    while ( v12 != v3 );
    result = a1;
    v3 = *(_QWORD *)a1;
  }
  if ( v3 != a1 + 16 )
    return _libc_free(v3, a2);
  return result;
}
