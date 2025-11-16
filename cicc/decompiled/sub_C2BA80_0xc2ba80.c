// Function: sub_C2BA80
// Address: 0xc2ba80
//
__int64 __fastcall sub_C2BA80(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 result; // rax
  __int64 v7; // rsi
  __int64 v8; // r14
  __int64 v9; // rsi
  __int64 v10; // [rsp+10h] [rbp-70h] BYREF
  __int64 v11; // [rsp+18h] [rbp-68h] BYREF
  __int64 v12; // [rsp+20h] [rbp-60h] BYREF
  char v13; // [rsp+30h] [rbp-50h]
  __int64 v14; // [rsp+40h] [rbp-40h] BYREF
  char v15; // [rsp+50h] [rbp-30h]

  a1[26] = a2;
  a1[27] = a3 + a2;
  sub_C21E40((__int64)&v12, a1);
  if ( (v13 & 1) == 0 || (result = (unsigned int)v12, !(_DWORD)v12) )
  {
    *a5 = v12;
    sub_C21E40((__int64)&v14, a1);
    if ( (v15 & 1) == 0 || (result = (unsigned int)v14, !(_DWORD)v14) )
    {
      if ( (unsigned __int8)sub_C5E690() )
      {
        v7 = *a5;
        v8 = a1[38];
        a1[48] += *a5;
        if ( a1[39] >= (unsigned __int64)(v8 + v7) && v8 )
          a1[38] = v8 + v7;
        else
          v8 = sub_9D1E70((__int64)(a1 + 38), v7, v7, 0);
        v9 = a1[26];
        v10 = *a5;
        sub_409308(&v11, v9, v14, v8, &v10);
      }
      sub_C1AFD0();
      return 13;
    }
  }
  return result;
}
