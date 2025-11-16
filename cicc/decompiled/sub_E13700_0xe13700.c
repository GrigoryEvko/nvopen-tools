// Function: sub_E13700
// Address: 0xe13700
//
__int64 *__fastcall sub_E13700(__int64 a1, __int64 *a2)
{
  __int64 v2; // rdi
  __int64 *result; // rax

  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 24) + 32LL))(*(_QWORD *)(a1 + 24));
  v2 = *(_QWORD *)(a1 + 24);
  result = (__int64 *)(*(_BYTE *)(v2 + 9) & 0xC0);
  if ( (*(_BYTE *)(v2 + 9) & 0xC0) != 0x80 )
  {
    if ( (*(_BYTE *)(v2 + 9) & 0xC0) == 0 )
      return result;
    return sub_E12F20(a2, 1u, " ");
  }
  result = (__int64 *)(**(__int64 (__fastcall ***)(__int64, __int64 *))v2)(v2, a2);
  if ( !(_BYTE)result )
    return sub_E12F20(a2, 1u, " ");
  return result;
}
