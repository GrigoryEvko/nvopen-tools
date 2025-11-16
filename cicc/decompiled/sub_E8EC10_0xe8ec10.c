// Function: sub_E8EC10
// Address: 0xe8ec10
//
__int64 __fastcall sub_E8EC10(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // rdi
  _QWORD *v6; // r13
  __int64 result; // rax
  _QWORD *v8; // r12

  *(_QWORD *)a1 = &unk_49E3518;
  v3 = *(_QWORD *)(a1 + 88);
  if ( v3 != a1 + 104 )
    _libc_free(v3, a2);
  v4 = *(_QWORD *)(a1 + 56);
  if ( v4 )
  {
    a2 = *(_QWORD *)(a1 + 72) - v4;
    j_j___libc_free_0(v4, a2);
  }
  v5 = *(_QWORD *)(a1 + 24);
  if ( v5 != a1 + 40 )
  {
    a2 = *(_QWORD *)(a1 + 40) + 1LL;
    j_j___libc_free_0(v5, a2);
  }
  v6 = *(_QWORD **)(a1 + 8);
  result = 5LL * *(unsigned int *)(a1 + 16);
  v8 = &v6[5 * *(unsigned int *)(a1 + 16)];
  if ( v6 != v8 )
  {
    do
    {
      v8 -= 5;
      result = (__int64)(v8 + 2);
      if ( (_QWORD *)*v8 != v8 + 2 )
      {
        a2 = v8[2] + 1LL;
        result = j_j___libc_free_0(*v8, a2);
      }
    }
    while ( v6 != v8 );
    v8 = *(_QWORD **)(a1 + 8);
  }
  if ( v8 != (_QWORD *)(a1 + 24) )
    return _libc_free(v8, a2);
  return result;
}
