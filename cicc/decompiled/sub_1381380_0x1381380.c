// Function: sub_1381380
// Address: 0x1381380
//
_QWORD *__fastcall sub_1381380(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  _QWORD *result; // rax
  __int64 v4; // rdx
  __int64 v5; // [rsp+8h] [rbp-68h] BYREF
  _QWORD v6[2]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v7[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v8; // [rsp+30h] [rbp-40h]
  _QWORD v9[2]; // [rsp+40h] [rbp-30h] BYREF
  __int16 v10; // [rsp+50h] [rbp-20h]

  if ( !qword_4F98A08
    || (v1 = sub_1649960(a1),
        v9[1] = v2,
        v9[0] = v1,
        result = (_QWORD *)sub_16D20C0(v9, qword_4F98A00, qword_4F98A08, 0),
        result != (_QWORD *)-1LL) )
  {
    v10 = 257;
    v6[0] = sub_1649960(a1);
    v6[1] = v4;
    v8 = 1283;
    v7[0] = "cfg";
    v7[1] = v6;
    v5 = a1;
    return sub_1381320((__int64)&v5, (__int64)v7, 0, (__int64)v9, 0);
  }
  return result;
}
