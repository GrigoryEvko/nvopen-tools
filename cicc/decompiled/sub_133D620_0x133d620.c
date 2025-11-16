// Function: sub_133D620
// Address: 0x133d620
//
__int64 __fastcall sub_133D620(
        _BYTE *a1,
        __int64 a2,
        __int64 a3,
        const char *a4,
        unsigned __int64 *a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  char v12; // r8
  __int64 result; // rax
  char *v14; // rsi
  unsigned __int64 v15; // rbx
  __int64 (__fastcall *v16)(_BYTE *, __int64, unsigned __int64, __int64, __int64, __int64, __int64); // rax
  char *v19[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( byte_4F96BB0 || (v12 = sub_133AC20(a1), result = 11, !v12) )
  {
    result = sub_131D480((__int64)a1, v19, a2, a3);
    if ( !(_DWORD)result )
    {
      v14 = v19[0];
      if ( !v19[0] || *((_QWORD *)v19[0] + 4) )
        return 2;
      *a5 -= a3;
      result = sub_131E100((__int64)a1, v14, a4, v19, a2 + 8 * a3, a5);
      v15 = *a5 + a3;
      *a5 = v15;
      if ( !(_DWORD)result )
      {
        if ( v19[0] )
        {
          v16 = (__int64 (__fastcall *)(_BYTE *, __int64, unsigned __int64, __int64, __int64, __int64, __int64))*((_QWORD *)v19[0] + 4);
          if ( v16 )
            return v16(a1, a2, v15, a6, a7, a8, a9);
        }
        return 2;
      }
    }
  }
  return result;
}
