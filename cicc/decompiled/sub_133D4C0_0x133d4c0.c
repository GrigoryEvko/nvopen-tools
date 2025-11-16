// Function: sub_133D4C0
// Address: 0x133d4c0
//
__int64 __fastcall sub_133D4C0(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  char v10; // r8
  __int64 result; // rax
  __int64 (__fastcall *v12)(_BYTE *, __int64, __int64, __int64, __int64, __int64, __int64); // rax
  char *v14; // [rsp+18h] [rbp-38h] BYREF

  if ( byte_4F96BB0 || (v10 = sub_133AC20(a1), result = 11, !v10) )
  {
    result = sub_131D480((__int64)a1, &v14, a2, a3);
    if ( !(_DWORD)result )
    {
      if ( v14
        && (v12 = (__int64 (__fastcall *)(_BYTE *, __int64, __int64, __int64, __int64, __int64, __int64))*((_QWORD *)v14 + 4)) != 0 )
      {
        return v12(a1, a2, a3, a4, a5, a6, a7);
      }
      else
      {
        return 2;
      }
    }
  }
  return result;
}
