// Function: sub_E14710
// Address: 0xe14710
//
__int64 *__fastcall sub_E14710(__int64 a1, __int64 a2)
{
  _BYTE *v2; // rdx
  __int64 *result; // rax
  __int64 *v4; // rax

  if ( *(_QWORD *)(a1 + 16) > 3u )
  {
    ++*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 40);
    sub_E12F20((__int64 *)a2, *(_QWORD *)(a1 + 16), *(const void **)(a1 + 24));
    --*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 41);
  }
  v2 = *(_BYTE **)(a1 + 40);
  if ( *v2 != 110 )
  {
    result = sub_E12F20((__int64 *)a2, *(_QWORD *)(a1 + 32), v2);
    if ( *(_QWORD *)(a1 + 16) > 3u )
      return result;
    return sub_E12F20((__int64 *)a2, *(_QWORD *)(a1 + 16), *(const void **)(a1 + 24));
  }
  v4 = (__int64 *)sub_E14360(a2, 45);
  result = sub_E12F20(v4, *(_QWORD *)(a1 + 32) - 1LL, (const void *)(*(_QWORD *)(a1 + 40) + 1LL));
  if ( *(_QWORD *)(a1 + 16) <= 3u )
    return sub_E12F20((__int64 *)a2, *(_QWORD *)(a1 + 16), *(const void **)(a1 + 24));
  return result;
}
