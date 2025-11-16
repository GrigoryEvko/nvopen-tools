// Function: sub_2251620
// Address: 0x2251620
//
_QWORD *__fastcall sub_2251620(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        _QWORD *a4,
        __int64 a5,
        char a6,
        __int64 a7,
        _DWORD *a8,
        __int64 *a9)
{
  _QWORD *v11; // rax
  _QWORD *v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rbx
  int v16; // edx
  __int64 v17; // [rsp+8h] [rbp-68h]
  __int64 v20[7]; // [rsp+38h] [rbp-38h] BYREF

  v17 = sub_2243120((_QWORD *)(a7 + 208), (__int64)a2);
  v20[0] = (__int64)&unk_4FD67D8;
  if ( a6 )
    v11 = sub_224F550(a1, a2, a3, a4, a5, a7, a8, v20);
  else
    v11 = sub_2250530(a1, a2, a3, a4, a5, a7, a8, v20);
  v12 = v11;
  v13 = v20[0];
  v14 = *(_QWORD *)(v20[0] - 24);
  if ( v14 )
  {
    sub_2216A40(a9, *(_QWORD *)(v20[0] - 24), 0);
    if ( *(int *)(*a9 - 8) >= 0 )
      sub_22163D0((const wchar_t **)a9);
    (*(void (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v17 + 88LL))(v17, v20[0], v20[0] + v14, *a9);
    v13 = v20[0];
  }
  if ( (_UNKNOWN *)(v13 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v16 = _InterlockedExchangeAdd((volatile signed __int32 *)(v13 - 8), 0xFFFFFFFF);
    }
    else
    {
      v16 = *(_DWORD *)(v13 - 8);
      *(_DWORD *)(v13 - 8) = v16 - 1;
    }
    if ( v16 <= 0 )
      j_j___libc_free_0_1(v13 - 24);
  }
  return v12;
}
