// Function: sub_A72330
// Address: 0xa72330
//
unsigned __int64 __fastcall sub_A72330(__int64 a1, __int64 a2)
{
  unsigned __int64 result; // rax
  unsigned __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  bool v8; // zf
  unsigned __int64 v9; // [rsp+8h] [rbp-38h] BYREF
  __int64 v10; // [rsp+10h] [rbp-30h] BYREF
  _QWORD v11[5]; // [rsp+18h] [rbp-28h] BYREF

  result = sub_B2D7E0(a1, "min-legal-vector-width", 22);
  v9 = result;
  if ( result )
  {
    v10 = sub_B2D7E0(a2, "min-legal-vector-width", 22);
    if ( v10 )
    {
      v3 = 0;
      v4 = sub_A72240((__int64 *)&v9);
      if ( !(unsigned __int8)sub_C93C90(v4, v5, 0, v11) )
        v3 = v11[0];
      v6 = sub_A72240(&v10);
      v8 = (unsigned __int8)sub_C93C90(v6, v7, 0, v11) == 0;
      result = 0;
      if ( v8 )
        result = v11[0];
      if ( v3 < result )
        return sub_B2CDC0(a1, v10);
    }
    else
    {
      return sub_B2D4A0(a1, "min-legal-vector-width", 22);
    }
  }
  return result;
}
