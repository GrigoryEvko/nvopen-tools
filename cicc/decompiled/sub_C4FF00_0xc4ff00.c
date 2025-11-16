// Function: sub_C4FF00
// Address: 0xc4ff00
//
__int64 __fastcall sub_C4FF00(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        char *a5,
        size_t a6,
        unsigned __int8 a7)
{
  __int64 v7; // r11
  size_t v9; // r14
  _BYTE *v12; // rax
  size_t v13; // rbx
  size_t v14; // r9
  __int64 result; // rax
  size_t v16; // rbx
  __int64 v17; // [rsp+8h] [rbp-38h]

  v7 = a1;
  v9 = a6;
  if ( (*(_BYTE *)(a1 + 13) & 2) != 0 && a6 )
  {
    while ( 1 )
    {
      v17 = v7;
      v12 = memchr(a5, 44, v9);
      v7 = v17;
      if ( !v12 )
        break;
      v13 = v12 - a5;
      if ( v12 - a5 == -1 )
        break;
      v14 = v12 - a5;
      if ( v9 <= v13 )
        v14 = v9;
      result = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64, char *, size_t, _QWORD))(*(_QWORD *)v17 + 80LL))(
                 v17,
                 a2,
                 a3,
                 a4,
                 a5,
                 v14,
                 a7);
      if ( (_BYTE)result )
        return result;
      v16 = v13 + 1;
      v7 = v17;
      if ( v16 > v9 )
      {
        a5 += v9;
LABEL_12:
        v9 = 0;
        return (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64, char *, size_t))(*(_QWORD *)v7 + 80LL))(
                 v7,
                 a2,
                 a3,
                 a4,
                 a5,
                 v9);
      }
      v9 -= v16;
      a5 += v16;
      if ( v9 != -1 && !v9 )
        goto LABEL_12;
    }
  }
  return (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64, char *, size_t))(*(_QWORD *)v7 + 80LL))(
           v7,
           a2,
           a3,
           a4,
           a5,
           v9);
}
