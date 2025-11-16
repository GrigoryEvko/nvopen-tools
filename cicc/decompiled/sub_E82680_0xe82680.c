// Function: sub_E82680
// Address: 0xe82680
//
__int64 __fastcall sub_E82680(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 result; // rax

  v2 = *(_QWORD *)(a1 + 64);
  if ( v2 != a1 + 80 )
    return _libc_free(v2, a2);
  return result;
}
