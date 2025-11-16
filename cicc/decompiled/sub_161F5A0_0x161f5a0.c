// Function: sub_161F5A0
// Address: 0x161f5a0
//
__int64 __fastcall sub_161F5A0(_QWORD *a1)
{
  __int64 v2; // r14
  __int64 v3; // r13
  __int64 v4; // r12
  __int64 v5; // rsi
  _QWORD *v6; // rdi
  __int64 result; // rax

  sub_161F550((__int64)a1);
  v2 = a1[7];
  if ( v2 )
  {
    v3 = *(_QWORD *)v2;
    v4 = *(_QWORD *)v2 + 8LL * *(unsigned int *)(v2 + 8);
    if ( *(_QWORD *)v2 != v4 )
    {
      do
      {
        v5 = *(_QWORD *)(v4 - 8);
        v4 -= 8;
        if ( v5 )
          sub_161E7C0(v4, v5);
      }
      while ( v3 != v4 );
      v4 = *(_QWORD *)v2;
    }
    if ( v4 != v2 + 16 )
      _libc_free(v4);
    j_j___libc_free_0(v2, 48);
  }
  v6 = (_QWORD *)a1[2];
  result = (__int64)(a1 + 4);
  if ( v6 != a1 + 4 )
    return j_j___libc_free_0(v6, a1[4] + 1LL);
  return result;
}
