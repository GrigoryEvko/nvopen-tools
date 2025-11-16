// Function: sub_38E3290
// Address: 0x38e3290
//
__int64 __fastcall sub_38E3290(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r14
  unsigned int v3; // r13d
  __int64 v5; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v6[2]; // [rsp+10h] [rbp-40h] BYREF
  char v7; // [rsp+20h] [rbp-30h]
  char v8; // [rsp+21h] [rbp-2Fh]

  v1 = sub_3909460();
  v2 = sub_39092A0(v1);
  if ( (unsigned __int8)sub_38E31C0(a1, &v5, (__int64)".cv_func_id", 11) )
    return 1;
  v8 = 1;
  v7 = 3;
  v6[0] = "unexpected token in '.cv_func_id' directive";
  v3 = sub_3909E20(a1, 9, v6);
  if ( (_BYTE)v3 )
  {
    return 1;
  }
  else if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 328) + 608LL))(
               *(_QWORD *)(a1 + 328),
               (unsigned int)v5) )
  {
    v8 = 1;
    v6[0] = "function id already allocated";
    v7 = 3;
    return (unsigned int)sub_3909790(a1, v2, v6, 0, 0);
  }
  return v3;
}
