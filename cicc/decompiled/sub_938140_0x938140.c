// Function: sub_938140
// Address: 0x938140
//
__int64 __fastcall sub_938140(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 result; // rax
  __int64 v7; // [rsp+8h] [rbp-98h] BYREF
  __int64 v8; // [rsp+10h] [rbp-90h] BYREF
  _BYTE *v9; // [rsp+18h] [rbp-88h]
  __int64 v10; // [rsp+20h] [rbp-80h]
  _BYTE v11[120]; // [rsp+28h] [rbp-78h] BYREF

  v4 = a2;
  v5 = *(_QWORD *)(a1 + 344);
  v9 = v11;
  v8 = v5;
  v10 = 0x800000000LL;
  sub_9374A0((__int64)&v8, a2);
  if ( (*(_BYTE *)(a2 + 196) & 0x40) != 0 )
  {
    a2 = 31;
    sub_A77B20(&v8, 31);
  }
  else
  {
    if ( (*(_BYTE *)(a2 + 202) & 1) != 0 || (*(_BYTE *)(a2 + 199) & 0x10) != 0 )
    {
      a2 = 3;
      sub_A77B20(&v8, 3);
    }
    if ( *(char *)(v4 + 192) < 0 )
    {
      a2 = 16;
      sub_A77B20(&v8, 16);
    }
  }
  result = (unsigned int)v10;
  if ( (_DWORD)v10 )
  {
    a2 = *(_QWORD *)(a1 + 344);
    v7 = *(_QWORD *)(a3 + 120);
    result = sub_A7B2C0(&v7, a2, 0xFFFFFFFFLL, &v8);
    *(_QWORD *)(a3 + 120) = result;
  }
  if ( v9 != v11 )
    return _libc_free(v9, a2);
  return result;
}
