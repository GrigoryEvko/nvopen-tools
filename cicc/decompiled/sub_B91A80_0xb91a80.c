// Function: sub_B91A80
// Address: 0xb91a80
//
__int64 __fastcall sub_B91A80(_QWORD *a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // r13
  __int64 v5; // r12
  _QWORD *v6; // rdi
  __int64 result; // rax

  sub_B91A30((__int64)a1);
  v3 = a1[7];
  if ( v3 )
  {
    v4 = *(_QWORD *)v3;
    v5 = *(_QWORD *)v3 + 8LL * *(unsigned int *)(v3 + 8);
    if ( *(_QWORD *)v3 != v5 )
    {
      do
      {
        a2 = *(_QWORD *)(v5 - 8);
        v5 -= 8;
        if ( a2 )
          sub_B91220(v5, a2);
      }
      while ( v4 != v5 );
      v5 = *(_QWORD *)v3;
    }
    if ( v5 != v3 + 16 )
      _libc_free(v5, a2);
    j_j___libc_free_0(v3, 48);
  }
  v6 = (_QWORD *)a1[2];
  result = (__int64)(a1 + 4);
  if ( v6 != a1 + 4 )
    return j_j___libc_free_0(v6, a1[4] + 1LL);
  return result;
}
