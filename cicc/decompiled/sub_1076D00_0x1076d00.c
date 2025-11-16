// Function: sub_1076D00
// Address: 0x1076d00
//
__int64 __fastcall sub_1076D00(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v3; // r8
  char v4; // al
  unsigned __int64 v5; // rsi
  unsigned int v7; // edx
  _BYTE *i; // rdi
  _BYTE *v10; // rcx
  int v11; // edx
  _BYTE v13[37]; // [rsp+Bh] [rbp-25h] BYREF

  v3 = 1;
  v4 = a2 & 0x7F;
  v5 = (unsigned __int64)a2 >> 7;
  v7 = 1;
  for ( i = v13; ; ++i )
  {
    v10 = i + 1;
    *i = v4 | 0x80;
    if ( !v5 )
      break;
    ++v7;
    v4 = v5 & 0x7F;
    LOBYTE(v3) = v7 <= 4;
    v5 >>= 7;
    if ( !v5 && v7 > 4 )
    {
      *v10 = v4;
      v11 = (_DWORD)i + 2;
      return (*(__int64 (__fastcall **)(__int64, _BYTE *, _QWORD, __int64, __int64))(*(_QWORD *)a1 + 104LL))(
               a1,
               v13,
               v11 - (unsigned int)v13,
               a3,
               v3);
    }
  }
  if ( (_BYTE)v3 )
  {
    if ( v7 != 4 )
      v10 = (char *)memset(i + 1, 128, 4 - v7) + 4 - v7;
    *v10 = 0;
    v11 = (_DWORD)v10 + 1;
  }
  else
  {
    v11 = (_DWORD)i + 1;
  }
  return (*(__int64 (__fastcall **)(__int64, _BYTE *, _QWORD, __int64, __int64))(*(_QWORD *)a1 + 104LL))(
           a1,
           v13,
           v11 - (unsigned int)v13,
           a3,
           v3);
}
