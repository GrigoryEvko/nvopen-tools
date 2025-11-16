// Function: sub_9C8010
// Address: 0x9c8010
//
__int64 __fastcall sub_9C8010(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rdi
  __int64 v5; // rdi

  result = a1 + 88;
  v4 = *(_QWORD *)(a1 + 72);
  if ( v4 != result )
    result = _libc_free(v4, a2);
  v5 = *(_QWORD *)(a1 + 8);
  if ( v5 != a1 + 24 )
    return _libc_free(v5, a2);
  return result;
}
