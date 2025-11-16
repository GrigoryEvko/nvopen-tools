// Function: sub_B2D810
// Address: 0xb2d810
//
__int64 __fastcall sub_B2D810(__int64 a1, const void *a2, size_t a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // [rsp+8h] [rbp-68h] BYREF
  _QWORD v11[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v12; // [rsp+30h] [rbp-40h]

  v10 = sub_B2D7E0(a1, a2, a3);
  if ( sub_A71840((__int64)&v10) )
  {
    v7 = sub_A72240(&v10);
    if ( (unsigned __int8)sub_C93C90(v7, v8, 0, v11) )
    {
      v9 = sub_B2BE50(a1);
      v11[2] = a2;
      v11[3] = a3;
      v12 = 1283;
      v11[0] = "cannot parse integer attribute ";
      sub_B6ECE0(v9, v11);
    }
    else
    {
      return v11[0];
    }
  }
  return a4;
}
