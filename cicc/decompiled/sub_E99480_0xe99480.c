// Function: sub_E99480
// Address: 0xe99480
//
__int64 __fastcall sub_E99480(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  _QWORD *v5; // rax
  _DWORD *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 result; // rax
  __int64 v10; // rdi
  const char *v11; // rax
  const char *v12; // [rsp+0h] [rbp-50h] BYREF
  char v13; // [rsp+20h] [rbp-30h]
  char v14; // [rsp+21h] [rbp-2Fh]

  v5 = sub_E66210(*(_QWORD *)(a1 + 8), a2);
  v6 = sub_E5F790((__int64)v5, a2);
  if ( !v6 )
  {
    v14 = 1;
    v10 = *(_QWORD *)(a1 + 8);
    v11 = "function id not introduced by .cv_func_id or .cv_inline_site_id";
LABEL_5:
    v12 = v11;
    v13 = 3;
    sub_E66880(v10, a4, (__int64)&v12);
    return 0;
  }
  v7 = *((_QWORD *)v6 + 2);
  v8 = *(_QWORD *)(*(_QWORD *)(a1 + 288) + 8LL);
  if ( v7 )
  {
    result = 1;
    if ( v7 == v8 )
      return result;
    v14 = 1;
    v10 = *(_QWORD *)(a1 + 8);
    v11 = "all .cv_loc directives for a function must be in the same section";
    goto LABEL_5;
  }
  *((_QWORD *)v6 + 2) = v8;
  return 1;
}
