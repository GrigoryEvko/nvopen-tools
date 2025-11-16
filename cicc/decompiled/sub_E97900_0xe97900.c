// Function: sub_E97900
// Address: 0xe97900
//
__int64 __fastcall sub_E97900(__int64 a1, __int64 a2, unsigned int a3, int a4, int a5, int a6, _QWORD *a7)
{
  _QWORD *v11; // rax
  _DWORD *v12; // rax
  __int64 v13; // rdi
  _QWORD *v14; // rax
  const char *v16; // [rsp+10h] [rbp-60h] BYREF
  char v17; // [rsp+30h] [rbp-40h]
  char v18; // [rsp+31h] [rbp-3Fh]

  v11 = sub_E66210(*(_QWORD *)(a1 + 8), a2);
  v12 = sub_E5F790((__int64)v11, a3);
  v13 = *(_QWORD *)(a1 + 8);
  if ( v12 )
  {
    v14 = sub_E66210(v13, a3);
    return sub_E626D0((__int64)v14, a2, a3, a4, a5, a6);
  }
  else
  {
    v18 = 1;
    v16 = "parent function id not introduced by .cv_func_id or .cv_inline_site_id";
    v17 = 3;
    sub_E66880(v13, a7, (__int64)&v16);
    return 1;
  }
}
