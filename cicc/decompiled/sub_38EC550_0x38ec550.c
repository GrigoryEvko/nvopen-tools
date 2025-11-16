// Function: sub_38EC550
// Address: 0x38ec550
//
__int64 __fastcall sub_38EC550(__int64 a1)
{
  __int64 v1; // r13
  __int64 v3; // [rsp+10h] [rbp-50h] BYREF
  __int64 v4; // [rsp+18h] [rbp-48h] BYREF
  _QWORD v5[2]; // [rsp+20h] [rbp-40h] BYREF
  char v6; // [rsp+30h] [rbp-30h]
  char v7; // [rsp+31h] [rbp-2Fh]

  v1 = sub_3909290(a1 + 144);
  if ( !*(_BYTE *)(a1 + 845) && (unsigned __int8)sub_38E36C0(a1) )
    return 1;
  v5[0] = 0;
  if ( sub_38EB6A0(a1, &v3, (__int64)v5) )
    return 1;
  v4 = 0;
  if ( (unsigned __int8)sub_3909EB0(a1, 25) && (unsigned __int8)sub_38EB9C0(a1, &v4)
    || (v7 = 1, v6 = 3, v5[0] = "unexpected token", (unsigned __int8)sub_3909E20(a1, 9, v5)) )
  {
    v7 = 1;
    v5[0] = " in '.org' directive";
    v6 = 3;
    return sub_39094A0(a1, v5);
  }
  else
  {
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD, __int64))(**(_QWORD **)(a1 + 328) + 528LL))(
      *(_QWORD *)(a1 + 328),
      v3,
      (unsigned __int8)v4,
      v1);
    return 0;
  }
}
