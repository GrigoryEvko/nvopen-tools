// Function: sub_122FE20
// Address: 0x122fe20
//
__int64 __fastcall sub_122FE20(__int64 **a1, __int64 *a2, __int64 *a3)
{
  __int64 result; // rax
  __int64 *v5; // [rsp+8h] [rbp-58h] BYREF
  const char *v6; // [rsp+10h] [rbp-50h] BYREF
  char v7; // [rsp+30h] [rbp-30h]
  char v8; // [rsp+31h] [rbp-2Fh]

  v5 = 0;
  v8 = 1;
  v6 = "expected type";
  v7 = 3;
  result = sub_12190A0((__int64)a1, &v5, (int *)&v6, 0);
  if ( !(_BYTE)result )
    return sub_1224B80(a1, (__int64)v5, a2, a3);
  return result;
}
