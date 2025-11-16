// Function: sub_1DA0810
// Address: 0x1da0810
//
__int64 __fastcall sub_1DA0810(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12
  unsigned __int64 v3; // rdi
  __int64 v4; // rsi
  __int64 result; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_1DA0810(v1[3]);
      v1 = (_QWORD *)v1[2];
      v3 = v2[11];
      if ( v3 != v2[10] )
        _libc_free(v3);
      v4 = v2[7];
      if ( v4 )
        sub_161E7C0((__int64)(v2 + 7), v4);
      result = j_j___libc_free_0(v2, 168);
    }
    while ( v1 );
  }
  return result;
}
