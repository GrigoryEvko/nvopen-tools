// Function: sub_120EA00
// Address: 0x120ea00
//
__int64 __fastcall sub_120EA00(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned int a7,
        unsigned int a8)
{
  __int64 result; // rax
  _QWORD v9[4]; // [rsp+0h] [rbp-C0h] BYREF
  __int16 v10; // [rsp+20h] [rbp-A0h]
  _QWORD v11[4]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v12; // [rsp+50h] [rbp-70h]
  _QWORD *v13; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v14; // [rsp+70h] [rbp-50h]
  __int16 v15; // [rsp+80h] [rbp-40h]
  _QWORD v16[4]; // [rsp+90h] [rbp-30h] BYREF
  __int16 v17; // [rsp+B0h] [rbp-10h]

  result = 0;
  if ( a8 < a7 )
  {
    v11[2] = a5;
    v10 = 773;
    v9[2] = " expected to be numbered '";
    v11[0] = v9;
    v9[0] = a3;
    v9[1] = a4;
    v13 = v11;
    v12 = 1282;
    v15 = 2306;
    v16[0] = &v13;
    v16[2] = "' or greater";
    v11[3] = a6;
    v14 = a7;
    v17 = 770;
    sub_11FD800(a1 + 176, a2, (__int64)v16, 1);
    return 1;
  }
  return result;
}
