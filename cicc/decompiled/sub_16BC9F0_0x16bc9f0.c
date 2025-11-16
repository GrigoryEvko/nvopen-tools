// Function: sub_16BC9F0
// Address: 0x16bc9f0
//
__int64 __fastcall sub_16BC9F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rdx
  __int64 *(__fastcall *v5)(__int64 *, __int64, int); // rax
  __int64 result; // rax
  const char *v7[2]; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v8[4]; // [rsp+10h] [rbp-20h] BYREF

  v3 = *(_QWORD *)(a1 + 16);
  v4 = *(unsigned int *)(a1 + 8);
  v5 = *(__int64 *(__fastcall **)(__int64 *, __int64, int))(*(_QWORD *)v3 + 32LL);
  if ( v5 == sub_16BC7D0 )
  {
    v7[0] = (const char *)v8;
    if ( (_DWORD)v4 == 1 )
      sub_16BC640((__int64 *)v7, "Multiple errors", (__int64)"");
    else
      sub_16BC640(
        (__int64 *)v7,
        "Inconvertible error value. An error has occurred that could not be converted to a known std::error_code. Please file a bug.",
        (__int64)"");
  }
  else
  {
    v5((__int64 *)v7, v3, v4);
  }
  result = sub_16E7EE0(a2, v7[0], v7[1]);
  if ( (_QWORD *)v7[0] != v8 )
    return j_j___libc_free_0(v7[0], v8[0] + 1LL);
  return result;
}
