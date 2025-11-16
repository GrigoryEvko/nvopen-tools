// Function: sub_3157460
// Address: 0x3157460
//
__int64 __fastcall sub_3157460(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 (__fastcall *v3)(__int64, __int64, char); // rbx
  __int64 v4; // rax
  __int64 result; // rax
  _BYTE v6[17]; // [rsp+Fh] [rbp-11h] BYREF

  v2 = *a1;
  v6[0] = 0;
  v3 = *(__int64 (__fastcall **)(__int64, __int64, char))(v2 + 64);
  v4 = sub_B2BEC0(a2);
  if ( v3 != sub_31572C0 )
    return ((__int64 (__fastcall *)(__int64 *, __int64, __int64, _QWORD, _BYTE *))v3)(a1, v4, 4, 0, v6);
  result = a1[6];
  if ( !result )
    return a1[4];
  return result;
}
