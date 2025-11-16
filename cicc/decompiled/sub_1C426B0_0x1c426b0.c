// Function: sub_1C426B0
// Address: 0x1c426b0
//
_QWORD *__fastcall sub_1C426B0(_QWORD *a1)
{
  _QWORD *result; // rax
  __int64 i; // rdi
  _QWORD *v4; // rbx
  __int64 v5; // rsi
  __int64 v6; // rdi
  __int64 v7; // [rsp-8h] [rbp-18h]

  result = &unk_49F7B08;
  *a1 = &unk_49F7B08;
  for ( i = a1[2]; a1[1] != i; i = a1[2] )
  {
    v4 = *(_QWORD **)(i - 8);
    v5 = *((unsigned int *)v4 + 2);
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))(*(_QWORD *)*v4 + 104LL))(
      *v4,
      v5,
      *((unsigned int *)v4 + 3),
      v4[2],
      v4[3],
      *((unsigned __int8 *)v4 + 48),
      v4[4],
      v4[5]);
    if ( *((_BYTE *)v4 + 48) )
      (*(void (__fastcall **)(_QWORD, __int64, __int64))(*(_QWORD *)*v4 + 96LL))(*v4, v5, v7);
    result = (_QWORD *)a1[2];
    a1[2] = result - 1;
    v6 = *(result - 1);
    if ( v6 )
      result = (_QWORD *)j_j___libc_free_0(v6, 56);
  }
  if ( i )
    return (_QWORD *)j_j___libc_free_0(i, a1[3] - i);
  return result;
}
