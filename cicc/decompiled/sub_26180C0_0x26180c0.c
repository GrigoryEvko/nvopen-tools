// Function: sub_26180C0
// Address: 0x26180c0
//
__int64 __fastcall sub_26180C0(__int64 a1, __int64 *a2)
{
  char v2; // r8
  __int64 result; // rax
  int v4; // eax
  unsigned __int8 v5; // [rsp+Fh] [rbp-71h] BYREF
  __int64 v6; // [rsp+10h] [rbp-70h] BYREF
  __int64 v7; // [rsp+18h] [rbp-68h] BYREF
  _QWORD v8[2]; // [rsp+20h] [rbp-60h] BYREF
  int v9; // [rsp+30h] [rbp-50h] BYREF
  __int64 (__fastcall *v10)(__int64 *, unsigned __int64); // [rsp+38h] [rbp-48h]
  __int64 *v11; // [rsp+40h] [rbp-40h]
  __int64 (__fastcall *v12)(__int64 *, unsigned __int64); // [rsp+48h] [rbp-38h]
  _QWORD *v13; // [rsp+50h] [rbp-30h]
  __int64 (__fastcall *v14)(__int64, __int64); // [rsp+58h] [rbp-28h]
  __int64 *v15; // [rsp+60h] [rbp-20h]

  v2 = sub_BB92D0(a1, a2);
  result = 0;
  if ( !v2 )
  {
    v5 = 0;
    v8[1] = &v5;
    v4 = *(_DWORD *)(a1 + 172);
    v6 = a1;
    v9 = v4;
    v10 = sub_26175F0;
    v11 = &v6;
    v12 = sub_2617590;
    v13 = v8;
    v14 = sub_2617880;
    v8[0] = a1;
    v7 = a1;
    v15 = &v7;
    result = sub_2617DD0((__int64)&v9, (__int64)a2);
    if ( !(_BYTE)result )
      return v5;
  }
  return result;
}
