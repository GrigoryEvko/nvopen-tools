// Function: sub_C82340
// Address: 0xc82340
//
__int64 __fastcall sub_C82340(__int64 a1, char a2, __mode_t a3)
{
  __int64 v5; // rsi
  const char *v6; // rdi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  unsigned int v10; // r14d
  _QWORD v12[3]; // [rsp+0h] [rbp-C0h] BYREF
  _BYTE v13[168]; // [rsp+18h] [rbp-A8h] BYREF

  v12[0] = v13;
  v12[1] = 0;
  v12[2] = 128;
  v5 = a3;
  v6 = (const char *)sub_CA12A0(a1, v12);
  if ( mkdir(v6, a3) != -1 || (v10 = *__errno_location(), v10 == 17) && a2 )
  {
    v10 = 0;
    sub_2241E40(v6, v5, v7, v8, v9);
  }
  else
  {
    sub_2241E50(v6, v5, v7, v8, v9);
  }
  if ( (_BYTE *)v12[0] != v13 )
    _libc_free(v12[0], v5);
  return v10;
}
