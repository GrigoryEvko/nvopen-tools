// Function: sub_2251510
// Address: 0x2251510
//
_QWORD *__fastcall sub_2251510(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        _QWORD *a4,
        __int64 a5,
        char a6,
        __int64 a7,
        _DWORD *a8,
        __int64 a9)
{
  _QWORD *v9; // rax
  _QWORD *v10; // r15
  unsigned __int64 v11; // rdi
  int v13; // edx
  volatile signed __int32 *v14; // [rsp+10h] [rbp-40h] BYREF
  __int64 v15[7]; // [rsp+18h] [rbp-38h] BYREF

  v14 = (volatile signed __int32 *)&unk_4FD67D8;
  if ( a6 )
    v9 = sub_224F550(a1, a2, a3, a4, a5, a7, a8, (__int64 *)&v14);
  else
    v9 = sub_2250530(a1, a2, a3, a4, a5, a7, a8, (__int64 *)&v14);
  v10 = v9;
  v15[0] = sub_2208E60(a1, v9);
  sub_2254180(v14, a9, a8, v15);
  v11 = (unsigned __int64)(v14 - 6);
  if ( v14 - 6 != (volatile signed __int32 *)&unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v13 = _InterlockedExchangeAdd(v14 - 2, 0xFFFFFFFF);
    }
    else
    {
      v13 = *((_DWORD *)v14 - 2);
      *((_DWORD *)v14 - 2) = v13 - 1;
    }
    if ( v13 <= 0 )
      j_j___libc_free_0_1(v11);
  }
  return v10;
}
