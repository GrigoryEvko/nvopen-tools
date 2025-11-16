// Function: sub_133D570
// Address: 0x133d570
//
__int64 __fastcall sub_133D570(_BYTE *a1, __int64 a2, __int64 a3, const char *a4, unsigned __int64 *a5)
{
  char v9; // r8
  __int64 result; // rax
  char *v11; // rsi
  char *v12; // [rsp+8h] [rbp-38h] BYREF

  if ( byte_4F96BB0 || (v9 = sub_133AC20(a1), result = 11, !v9) )
  {
    result = sub_131D480((__int64)a1, &v12, a2, a3);
    if ( !(_DWORD)result )
    {
      v11 = v12;
      result = 2;
      if ( v12 )
      {
        if ( !*((_QWORD *)v12 + 4) )
        {
          *a5 -= a3;
          result = sub_131E100((__int64)a1, v11, a4, 0, a2 + 8 * a3, a5);
          *a5 += a3;
        }
      }
    }
  }
  return result;
}
