// Function: sub_10E21D0
// Address: 0x10e21d0
//
__int64 __fastcall sub_10E21D0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  _QWORD *v4; // rbx
  __int64 result; // rax
  _QWORD *v6; // r12
  __int64 v7; // rdi

  v3 = *(unsigned int *)(a1 + 8);
  v4 = *(_QWORD **)a1;
  result = 7 * v3;
  v6 = (_QWORD *)(*(_QWORD *)a1 + 56 * v3);
  if ( *(_QWORD **)a1 != v6 )
  {
    do
    {
      v7 = *(v6 - 3);
      v6 -= 7;
      if ( v7 )
      {
        a2 = v6[6] - v7;
        j_j___libc_free_0(v7, a2);
      }
      result = (__int64)(v6 + 2);
      if ( (_QWORD *)*v6 != v6 + 2 )
      {
        a2 = v6[2] + 1LL;
        result = j_j___libc_free_0(*v6, a2);
      }
    }
    while ( v4 != v6 );
    v6 = *(_QWORD **)a1;
  }
  if ( v6 != (_QWORD *)(a1 + 16) )
    return _libc_free(v6, a2);
  return result;
}
