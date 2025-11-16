// Function: sub_D89A50
// Address: 0xd89a50
//
__int64 (__fastcall *__fastcall sub_D89A50(__int64 a1))(__int64, __int64, __int64)
{
  __int64 v1; // r12
  __int64 (__fastcall *result)(__int64, __int64, __int64); // rax

  v1 = *(_QWORD *)(a1 + 40);
  if ( v1 )
  {
    sub_D85F30(*(_QWORD **)(v1 + 64));
    sub_D85E30(*(_QWORD **)(v1 + 16));
    j_j___libc_free_0(v1, 104);
  }
  result = *(__int64 (__fastcall **)(__int64, __int64, __int64))(a1 + 24);
  if ( result )
    return (__int64 (__fastcall *)(__int64, __int64, __int64))result(a1 + 8, a1 + 8, 3);
  return result;
}
