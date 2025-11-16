// Function: sub_9C7CD0
// Address: 0x9c7cd0
//
_QWORD *__fastcall sub_9C7CD0(__int64 a1, __int64 *a2)
{
  _QWORD *result; // rax
  __int64 v3; // rdx
  __int64 v4; // rdx
  __int64 v5; // rdx
  _QWORD *v6; // r13

  result = (_QWORD *)sub_22077B0(24);
  if ( result )
  {
    v3 = *a2;
    *a2 = 0;
    *result = v3;
    v4 = a2[1];
    a2[1] = 0;
    result[1] = v4;
    v5 = a2[2];
    a2[2] = 0;
    result[2] = v5;
  }
  v6 = *(_QWORD **)(a1 + 56);
  *(_QWORD *)(a1 + 56) = result;
  if ( v6 )
  {
    if ( *v6 )
      j_j___libc_free_0(*v6, v6[2] - *v6);
    return (_QWORD *)j_j___libc_free_0(v6, 24);
  }
  return result;
}
