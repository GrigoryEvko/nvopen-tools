// Function: sub_B7CDF0
// Address: 0xb7cdf0
//
_BYTE *__fastcall sub_B7CDF0(__int64 a1, __int64 a2)
{
  size_t v2; // rdx
  __int64 v3; // r13
  __int64 v4; // r14
  _BYTE *result; // rax
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  _BYTE v9[104]; // [rsp+0h] [rbp-1E0h] BYREF
  _BYTE *v10; // [rsp+68h] [rbp-178h]
  _BYTE v11[360]; // [rsp+78h] [rbp-168h] BYREF

  v2 = 0;
  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)a1;
  if ( v3 )
    v2 = strlen(*(const char **)(a2 + 40));
  result = (_BYTE *)sub_C2EF00(v4, v3, v2);
  if ( (_BYTE)result )
  {
    sub_B7C9C0((__int64)v9, a1, (const __m128i *)a2, v6, v7, v8);
    (*(void (__fastcall **)(_QWORD, _BYTE *))(**(_QWORD **)(*(_QWORD *)a1 + 24LL) + 16LL))(
      *(_QWORD *)(*(_QWORD *)a1 + 24LL),
      v9);
    result = v11;
    if ( v10 != v11 )
      return (_BYTE *)_libc_free(v10, v9);
  }
  return result;
}
