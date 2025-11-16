// Function: sub_C5FAF0
// Address: 0xc5faf0
//
__int64 (__fastcall *__fastcall sub_C5FAF0(__int64 a1, __int64 a2))(__int64, __int64, __int64)
{
  __int64 (__fastcall *result)(__int64, __int64, __int64); // rax
  __int64 v4; // rdi
  _QWORD *v5; // rbx
  _QWORD *v6; // r12
  _QWORD *v7; // rdi
  __int64 v8; // rdi

  *(_QWORD *)a1 = &unk_49DC600;
  result = *(__int64 (__fastcall **)(__int64, __int64, __int64))(a1 + 224);
  if ( result )
  {
    a2 = a1 + 208;
    result = (__int64 (__fastcall *)(__int64, __int64, __int64))result(a2, a2, 3);
  }
  v4 = *(_QWORD *)(a1 + 176);
  if ( v4 )
  {
    a2 = *(_QWORD *)(a1 + 192) - v4;
    result = (__int64 (__fastcall *)(__int64, __int64, __int64))j_j___libc_free_0(v4, a2);
  }
  v5 = *(_QWORD **)(a1 + 152);
  v6 = *(_QWORD **)(a1 + 144);
  if ( v5 != v6 )
  {
    do
    {
      v7 = (_QWORD *)v6[1];
      result = (__int64 (__fastcall *)(__int64, __int64, __int64))(v6 + 3);
      *v6 = &unk_49DACE8;
      if ( v7 != v6 + 3 )
      {
        a2 = v6[3] + 1LL;
        result = (__int64 (__fastcall *)(__int64, __int64, __int64))j_j___libc_free_0(v7, a2);
      }
      v6 += 6;
    }
    while ( v5 != v6 );
    v6 = *(_QWORD **)(a1 + 144);
  }
  if ( v6 )
  {
    a2 = *(_QWORD *)(a1 + 160) - (_QWORD)v6;
    result = (__int64 (__fastcall *)(__int64, __int64, __int64))j_j___libc_free_0(v6, a2);
  }
  if ( !*(_BYTE *)(a1 + 124) )
    result = (__int64 (__fastcall *)(__int64, __int64, __int64))_libc_free(*(_QWORD *)(a1 + 104), a2);
  v8 = *(_QWORD *)(a1 + 72);
  if ( v8 != a1 + 88 )
    return (__int64 (__fastcall *)(__int64, __int64, __int64))_libc_free(v8, a2);
  return result;
}
