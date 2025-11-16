// Function: sub_1224A40
// Address: 0x1224a40
//
__int64 __fastcall sub_1224A40(__int64 **a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 *v3; // [rsp+8h] [rbp-48h] BYREF
  const char *v4; // [rsp+10h] [rbp-40h] BYREF
  char v5; // [rsp+30h] [rbp-20h]
  char v6; // [rsp+31h] [rbp-1Fh]

  v3 = 0;
  v6 = 1;
  v4 = "expected type";
  v5 = 3;
  result = sub_12190A0((__int64)a1, &v3, (int *)&v4, 0);
  if ( !(_BYTE)result )
    return sub_1224770(a1, (__int64)v3, a2);
  return result;
}
