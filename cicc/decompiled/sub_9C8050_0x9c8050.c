// Function: sub_9C8050
// Address: 0x9c8050
//
__int64 __fastcall sub_9C8050(_QWORD *a1, __int64 a2)
{
  __int64 *v3; // r13
  __int64 *v4; // r12
  __int64 v5; // rdi
  __int64 result; // rax
  __int64 v7; // r13
  __int64 v8; // r12
  __int64 v9; // rdi

  v3 = (__int64 *)a1[12];
  v4 = (__int64 *)a1[11];
  if ( v3 != v4 )
  {
    do
    {
      v5 = *v4;
      if ( *v4 )
      {
        a2 = v4[2] - v5;
        result = j_j___libc_free_0(v5, a2);
      }
      v4 += 3;
    }
    while ( v3 != v4 );
    v4 = (__int64 *)a1[11];
  }
  if ( v4 )
  {
    a2 = a1[13] - (_QWORD)v4;
    result = j_j___libc_free_0(v4, a2);
  }
  v7 = a1[9];
  v8 = a1[8];
  if ( v7 != v8 )
  {
    do
    {
      v9 = *(_QWORD *)(v8 + 8);
      result = v8 + 24;
      if ( v9 != v8 + 24 )
        result = _libc_free(v9, a2);
      v8 += 72;
    }
    while ( v7 != v8 );
    v8 = a1[8];
  }
  if ( v8 )
  {
    a2 = a1[10] - v8;
    result = j_j___libc_free_0(v8, a2);
  }
  if ( (_QWORD *)*a1 != a1 + 3 )
    return _libc_free(*a1, a2);
  return result;
}
