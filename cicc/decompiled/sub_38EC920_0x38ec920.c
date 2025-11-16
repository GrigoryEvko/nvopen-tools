// Function: sub_38EC920
// Address: 0x38ec920
//
__int64 __fastcall sub_38EC920(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  _QWORD v5[2]; // [rsp+10h] [rbp-80h] BYREF
  __int64 v6; // [rsp+20h] [rbp-70h] BYREF
  __int64 v7; // [rsp+28h] [rbp-68h] BYREF
  _QWORD v8[2]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v9; // [rsp+40h] [rbp-50h]
  _QWORD v10[2]; // [rsp+50h] [rbp-40h] BYREF
  __int16 v11; // [rsp+60h] [rbp-30h]

  v5[0] = a2;
  v5[1] = a3;
  v3 = sub_3909290(a1 + 144);
  if ( !*(_BYTE *)(a1 + 845) && (unsigned __int8)sub_38E36C0(a1) )
    return 1;
  v10[0] = 0;
  if ( sub_38EB6A0(a1, &v6, (__int64)v10) )
    return 1;
  v7 = 0;
  if ( (unsigned __int8)sub_3909EB0(a1, 25) && (unsigned __int8)sub_38EB9C0(a1, &v7)
    || (v11 = 259, v10[0] = "unexpected token", (unsigned __int8)sub_3909E20(a1, 9, v10)) )
  {
    v8[0] = "in '";
    v8[1] = v5;
    v9 = 1283;
    v10[0] = v8;
    v10[1] = "' directive";
    v11 = 770;
    return sub_39094A0(a1, v10);
  }
  else
  {
    (*(void (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a1 + 328) + 496LL))(
      *(_QWORD *)(a1 + 328),
      v6,
      v7,
      v3);
    return 0;
  }
}
