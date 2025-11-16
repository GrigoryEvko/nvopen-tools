// Function: sub_E98210
// Address: 0xe98210
//
void *__fastcall sub_E98210(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  void (__fastcall *v4)(__int64, __int64); // rax
  void *result; // rax
  __int64 v6; // rsi

  v2 = *(_QWORD *)(a1 + 120) + 32LL * *(unsigned int *)(a1 + 128) - 32;
  *(_QWORD *)(v2 + 16) = *(_QWORD *)v2;
  *(_DWORD *)(v2 + 24) = *(_DWORD *)(v2 + 8);
  v3 = *(_QWORD *)(a1 + 120) + 32LL * *(unsigned int *)(a1 + 128) - 32;
  *(_QWORD *)v3 = a2;
  *(_DWORD *)(v3 + 8) = 0;
  v4 = **(void (__fastcall ***)(__int64, __int64))a1;
  if ( v4 == sub_E97740 )
  {
    result = (void *)(a2 + 56);
    *(_QWORD *)(a1 + 288) = a2 + 56;
  }
  else
  {
    result = (void *)((__int64 (__fastcall *)(__int64, __int64, _QWORD))v4)(a1, a2, 0);
  }
  v6 = *(_QWORD *)(a2 + 16);
  if ( v6 )
  {
    result = *(void **)v6;
    if ( !*(_QWORD *)v6 )
    {
      if ( (*(_BYTE *)(v6 + 9) & 0x70) != 0x20 )
        return (void *)(*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 208LL))(a1, v6, 0);
      if ( *(char *)(v6 + 8) < 0 )
        return (void *)(*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 208LL))(a1, v6, 0);
      *(_BYTE *)(v6 + 8) |= 8u;
      result = sub_E807D0(*(_QWORD *)(v6 + 24));
      *(_QWORD *)v6 = result;
      if ( !result )
        return (void *)(*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 208LL))(a1, v6, 0);
    }
    if ( result == off_4C5D170 )
      return (void *)(*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 208LL))(a1, v6, 0);
  }
  return result;
}
