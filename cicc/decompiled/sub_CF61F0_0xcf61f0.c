// Function: sub_CF61F0
// Address: 0xcf61f0
//
__int64 __fastcall sub_CF61F0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 result; // rax
  _BYTE v7[104]; // [rsp-68h] [rbp-68h] BYREF

  if ( !*a3 )
    return 3;
  sub_D666C0(v7);
  result = sub_CF4D50(a1, (__int64)v7, (__int64)a3, a4, a2);
  if ( (_BYTE)result )
    return sub_CF4FA0(a1, (__int64)a3, a4, 0);
  return result;
}
