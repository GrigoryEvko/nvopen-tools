// Function: sub_38DBE80
// Address: 0x38dbe80
//
__int64 __fastcall sub_38DBE80(
        __int64 a1,
        unsigned int a2,
        unsigned int a3,
        unsigned int a4,
        unsigned int a5,
        unsigned int a6,
        unsigned __int64 a7)
{
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  const char *v16; // [rsp+10h] [rbp-50h] BYREF
  char v17; // [rsp+20h] [rbp-40h]
  char v18; // [rsp+21h] [rbp-3Fh]

  v11 = sub_38BE350(*(_QWORD *)(a1 + 8));
  v12 = sub_390FE40(v11, a3);
  v13 = *(_QWORD *)(a1 + 8);
  if ( v12 )
  {
    v14 = sub_38BE350(v13);
    return sub_3912DF0(v14, a2, a3, a4, a5, a6);
  }
  else
  {
    v18 = 1;
    v16 = "parent function id not introduced by .cv_func_id or .cv_inline_site_id";
    v17 = 3;
    sub_38BE3D0(v13, a7, (__int64)&v16);
    return 1;
  }
}
