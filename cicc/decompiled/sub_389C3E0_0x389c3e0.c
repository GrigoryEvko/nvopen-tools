// Function: sub_389C3E0
// Address: 0x389c3e0
//
__int64 __fastcall sub_389C3E0(__int64 **a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // [rsp+8h] [rbp-38h] BYREF
  const char *v4; // [rsp+10h] [rbp-30h] BYREF
  char v5; // [rsp+20h] [rbp-20h]
  char v6; // [rsp+21h] [rbp-1Fh]

  v3 = 0;
  v6 = 1;
  v4 = "expected type";
  v5 = 3;
  result = sub_3891B00((__int64)a1, &v3, (__int64)&v4, 0);
  if ( !(_BYTE)result )
    return sub_389C160(a1, v3, a2);
  return result;
}
