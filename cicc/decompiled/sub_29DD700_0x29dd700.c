// Function: sub_29DD700
// Address: 0x29dd700
//
__int64 __fastcall sub_29DD700(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  unsigned __int8 v7; // [rsp+Fh] [rbp-A1h]
  __int64 v8; // [rsp+10h] [rbp-A0h] BYREF
  char *v9; // [rsp+18h] [rbp-98h]
  __int64 v10; // [rsp+20h] [rbp-90h]
  int v11; // [rsp+28h] [rbp-88h]
  char v12; // [rsp+2Ch] [rbp-84h]
  char v13; // [rsp+30h] [rbp-80h] BYREF

  v8 = 0;
  v9 = &v13;
  v10 = 16;
  v11 = 0;
  v12 = 1;
  result = sub_29DD190(a1, a2, (__int64)&v8, a4, a5, a6);
  if ( !v12 )
  {
    v7 = result;
    _libc_free((unsigned __int64)v9);
    return v7;
  }
  return result;
}
