// Function: sub_E77F70
// Address: 0xe77f70
//
__int64 __fastcall sub_E77F70(_QWORD *a1, int a2, __int64 a3, unsigned __int64 a4)
{
  __int64 v4; // r9
  __int64 v5; // rsi
  __int64 result; // rax
  _BYTE *v7; // [rsp+10h] [rbp-130h] BYREF
  __int64 v8; // [rsp+18h] [rbp-128h]
  __int64 v9; // [rsp+20h] [rbp-120h]
  _BYTE v10[280]; // [rsp+28h] [rbp-118h] BYREF

  v4 = a1[1];
  v7 = v10;
  v8 = 0;
  v9 = 256;
  sub_E77860(v4, (unsigned __int16)a2 | (BYTE2(a2) << 16), a3, a4, (__int64 *)&v7);
  v5 = (__int64)v7;
  result = (*(__int64 (__fastcall **)(_QWORD *, _BYTE *, __int64))(*a1 + 512LL))(a1, v7, v8);
  if ( v7 != v10 )
    return _libc_free(v7, v5);
  return result;
}
