// Function: sub_133D380
// Address: 0x133d380
//
__int64 __fastcall sub_133D380(_BYTE *a1, const char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v9; // r8
  __int64 result; // rax
  __int64 (__fastcall *v11)(_BYTE *, _BYTE *, unsigned __int64, __int64, __int64, __int64, __int64); // rax
  unsigned __int64 v13; // [rsp+10h] [rbp-80h] BYREF
  __int64 v14; // [rsp+18h] [rbp-78h] BYREF
  _BYTE v15[112]; // [rsp+20h] [rbp-70h] BYREF

  if ( byte_4F96BB0 || (v9 = sub_133AC20(a1), result = 11, !v9) )
  {
    v13 = 7;
    result = sub_131E100((__int64)a1, qword_497FB60, a2, &v14, (__int64)v15, &v13);
    if ( !(_DWORD)result )
    {
      if ( v14
        && (v11 = *(__int64 (__fastcall **)(_BYTE *, _BYTE *, unsigned __int64, __int64, __int64, __int64, __int64))(v14 + 32)) != 0 )
      {
        return v11(a1, v15, v13, a3, a4, a5, a6);
      }
      else
      {
        return 2;
      }
    }
  }
  return result;
}
