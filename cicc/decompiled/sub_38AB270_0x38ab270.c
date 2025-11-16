// Function: sub_38AB270
// Address: 0x38ab270
//
__int64 __fastcall sub_38AB270(__int64 **a1, _QWORD *a2, __int64 *a3, double a4, double a5, double a6)
{
  __int64 result; // rax
  __int64 v8; // [rsp+8h] [rbp-48h] BYREF
  const char *v9; // [rsp+10h] [rbp-40h] BYREF
  char v10; // [rsp+20h] [rbp-30h]
  char v11; // [rsp+21h] [rbp-2Fh]

  v8 = 0;
  v11 = 1;
  v9 = "expected type";
  v10 = 3;
  result = sub_3891B00((__int64)a1, &v8, (__int64)&v9, 0);
  if ( !(_BYTE)result )
    return sub_38A1070(a1, v8, a2, a3, a4, a5, a6);
  return result;
}
