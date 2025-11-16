// Function: sub_1F34230
// Address: 0x1f34230
//
__int64 __fastcall sub_1F34230(__int64 *a1, __int64 a2)
{
  __int64 *v3; // r12
  __int64 *v4; // r14
  __int64 v5; // rsi
  __int64 v6; // rdi
  __int64 (*v7)(); // rax
  __int64 v9; // [rsp+0h] [rbp-F0h] BYREF
  __int64 v10; // [rsp+8h] [rbp-E8h] BYREF
  _BYTE *v11; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v12; // [rsp+18h] [rbp-D8h]
  _BYTE v13[208]; // [rsp+20h] [rbp-D0h] BYREF

  v3 = *(__int64 **)(a2 + 72);
  v4 = *(__int64 **)(a2 + 64);
  if ( v4 == v3 )
    return 1;
  while ( 1 )
  {
    v5 = *v4;
    if ( (unsigned int)((__int64)(*(_QWORD *)(*v4 + 96) - *(_QWORD *)(*v4 + 88)) >> 3) > 1 )
      break;
    v6 = *a1;
    v9 = 0;
    v10 = 0;
    v11 = v13;
    v12 = 0x400000000LL;
    v7 = *(__int64 (**)())(*(_QWORD *)v6 + 264LL);
    if ( v7 == sub_1D820E0 )
      break;
    if ( ((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *, __int64 *, _BYTE **, _QWORD))v7)(
           v6,
           v5,
           &v9,
           &v10,
           &v11,
           0)
      || (_DWORD)v12 )
    {
      if ( v11 != v13 )
        _libc_free((unsigned __int64)v11);
      return 0;
    }
    if ( v11 != v13 )
      _libc_free((unsigned __int64)v11);
    if ( v3 == ++v4 )
      return 1;
  }
  return 0;
}
