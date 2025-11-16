// Function: sub_B15200
// Address: 0xb15200
//
__int64 __fastcall sub_B15200(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rbx
  __int64 result; // rax
  _QWORD *v5; // r12
  _QWORD *v6; // rdi

  v3 = *(_QWORD **)(a1 + 80);
  *(_QWORD *)a1 = &unk_49D9D40;
  result = *(unsigned int *)(a1 + 88);
  v5 = &v3[10 * result];
  if ( v3 != v5 )
  {
    do
    {
      v5 -= 10;
      v6 = (_QWORD *)v5[4];
      if ( v6 != v5 + 6 )
      {
        a2 = v5[6] + 1LL;
        j_j___libc_free_0(v6, a2);
      }
      result = (__int64)(v5 + 2);
      if ( (_QWORD *)*v5 != v5 + 2 )
      {
        a2 = v5[2] + 1LL;
        result = j_j___libc_free_0(*v5, a2);
      }
    }
    while ( v3 != v5 );
    v5 = *(_QWORD **)(a1 + 80);
  }
  if ( v5 != (_QWORD *)(a1 + 96) )
    return _libc_free(v5, a2);
  return result;
}
