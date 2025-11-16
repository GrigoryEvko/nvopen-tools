// Function: sub_B81E70
// Address: 0xb81e70
//
__int64 __fastcall sub_B81E70(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rbx
  _QWORD *i; // r13
  __int64 v5; // rdi
  __int64 v6; // rsi
  __int64 v7; // rdi
  __int64 v8; // r12
  __int64 v9; // rsi
  __int64 result; // rax
  __int64 v11; // rdi

  v3 = *(_QWORD **)(a1 + 16);
  *(_QWORD *)a1 = &unk_49DA9F0;
  for ( i = &v3[*(unsigned int *)(a1 + 24)]; i != v3; ++v3 )
  {
    if ( *v3 )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v3 + 8LL))(*v3);
  }
  v5 = *(_QWORD *)(a1 + 240);
  if ( v5 != a1 + 256 )
    _libc_free(v5, a2);
  v6 = *(unsigned int *)(a1 + 232);
  v7 = *(_QWORD *)(a1 + 216);
  v8 = a1 + 32;
  v9 = 16 * v6;
  result = sub_C7D6A0(v7, v9, 8);
  v11 = *(_QWORD *)(v8 - 16);
  if ( v11 != v8 )
    return _libc_free(v11, v9);
  return result;
}
