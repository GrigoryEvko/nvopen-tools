// Function: sub_E980F0
// Address: 0xe980f0
//
_DWORD *__fastcall sub_E980F0(__int64 a1, __int64 a2, unsigned int a3)
{
  _DWORD *result; // rax
  __int64 v5; // rcx
  int v6; // edx
  void (__fastcall *v7)(__int64, __int64); // rax
  __int64 v8; // rsi

  result = (_DWORD *)(*(_QWORD *)(a1 + 120) + 32LL * *(unsigned int *)(a1 + 128) - 32);
  v5 = *(_QWORD *)result;
  v6 = result[2];
  *((_QWORD *)result + 2) = *(_QWORD *)result;
  result[6] = v6;
  if ( v6 != a3 || v5 != a2 )
  {
    v7 = **(void (__fastcall ***)(__int64, __int64))a1;
    if ( v7 == sub_E97740 )
      *(_QWORD *)(a1 + 288) = a2 + 56;
    else
      ((void (__fastcall *)(__int64, __int64, _QWORD))v7)(a1, a2, a3);
    result = (_DWORD *)(*(_QWORD *)(a1 + 120) + 32LL * *(unsigned int *)(a1 + 128) - 32);
    *(_QWORD *)result = a2;
    result[2] = a3;
    v8 = *(_QWORD *)(a2 + 16);
    if ( v8 )
    {
      result = *(_DWORD **)v8;
      if ( !*(_QWORD *)v8 )
      {
        if ( (*(_BYTE *)(v8 + 9) & 0x70) != 0x20 )
          return (_DWORD *)(*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 208LL))(a1, v8, 0);
        if ( *(char *)(v8 + 8) < 0 )
          return (_DWORD *)(*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 208LL))(a1, v8, 0);
        *(_BYTE *)(v8 + 8) |= 8u;
        result = sub_E807D0(*(_QWORD *)(v8 + 24));
        *(_QWORD *)v8 = result;
        if ( !result )
          return (_DWORD *)(*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 208LL))(a1, v8, 0);
      }
      if ( off_4C5D170 == (_UNKNOWN *)result )
        return (_DWORD *)(*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 208LL))(a1, v8, 0);
    }
  }
  return result;
}
