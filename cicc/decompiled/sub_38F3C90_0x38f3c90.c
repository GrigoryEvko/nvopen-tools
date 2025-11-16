// Function: sub_38F3C90
// Address: 0x38f3c90
//
__int64 __fastcall sub_38F3C90(__int64 a1)
{
  __int64 v2; // rax
  unsigned int v3; // r12d
  __int64 v5; // rdx
  unsigned int v6; // ecx
  unsigned __int8 v7; // al
  __int64 v8; // rdx
  unsigned int v9; // ecx
  unsigned __int8 v10; // al
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // [rsp+0h] [rbp-130h] BYREF
  __int64 v17; // [rsp+8h] [rbp-128h] BYREF
  unsigned __int64 v18; // [rsp+10h] [rbp-120h] BYREF
  __int64 v19; // [rsp+18h] [rbp-118h] BYREF
  __int64 v20[2]; // [rsp+20h] [rbp-110h] BYREF
  __int64 v21[2]; // [rsp+30h] [rbp-100h] BYREF
  const char *v22; // [rsp+40h] [rbp-F0h] BYREF
  char v23; // [rsp+50h] [rbp-E0h]
  char v24; // [rsp+51h] [rbp-DFh]
  const char *v25; // [rsp+60h] [rbp-D0h] BYREF
  char v26; // [rsp+70h] [rbp-C0h]
  char v27; // [rsp+71h] [rbp-BFh]
  const char *v28; // [rsp+80h] [rbp-B0h] BYREF
  char v29; // [rsp+90h] [rbp-A0h]
  char v30; // [rsp+91h] [rbp-9Fh]
  const char *v31; // [rsp+A0h] [rbp-90h] BYREF
  char v32; // [rsp+B0h] [rbp-80h]
  char v33; // [rsp+B1h] [rbp-7Fh]
  const char *v34; // [rsp+C0h] [rbp-70h] BYREF
  char v35; // [rsp+D0h] [rbp-60h]
  char v36; // [rsp+D1h] [rbp-5Fh]
  _QWORD v37[2]; // [rsp+E0h] [rbp-50h] BYREF
  __int16 v38; // [rsp+F0h] [rbp-40h]

  v20[0] = 0;
  v20[1] = 0;
  v21[0] = 0;
  v21[1] = 0;
  v2 = sub_3909460(a1);
  v19 = sub_39092A0(v2);
  if ( (unsigned __int8)sub_38E31C0(a1, &v16, (__int64)".cv_inline_linetable", 20) )
    return 1;
  if ( (unsigned __int8)sub_3909470(a1, &v19) )
    return 1;
  v24 = 1;
  v22 = "expected SourceField in '.cv_inline_linetable' directive";
  v23 = 3;
  if ( (unsigned __int8)sub_3909D40(a1, &v17, &v22) )
    return 1;
  v27 = 1;
  v25 = "File id less than zero in '.cv_inline_linetable' directive";
  v26 = 3;
  if ( (unsigned __int8)sub_3909C80(a1, v17 <= 0, v19, &v25) )
    return 1;
  if ( (unsigned __int8)sub_3909470(a1, &v19) )
    return 1;
  v30 = 1;
  v28 = "expected SourceLineNum in '.cv_inline_linetable' directive";
  v29 = 3;
  if ( (unsigned __int8)sub_3909D40(a1, &v18, &v28) )
    return 1;
  v33 = 1;
  v31 = "Line number less than zero in '.cv_inline_linetable' directive";
  v32 = 3;
  if ( (unsigned __int8)sub_3909C80(a1, v18 >> 63, v19, &v31) )
    return 1;
  if ( (unsigned __int8)sub_3909470(a1, &v19) )
    return 1;
  v36 = 1;
  v34 = "expected identifier in directive";
  v35 = 3;
  v7 = sub_38F0EE0(a1, v20, v5, v6);
  if ( (unsigned __int8)sub_3909C80(a1, v7, v19, &v34) )
    return 1;
  if ( (unsigned __int8)sub_3909470(a1, &v19) )
    return 1;
  v37[0] = "expected identifier in directive";
  v38 = 259;
  v10 = sub_38F0EE0(a1, v21, v8, v9);
  if ( (unsigned __int8)sub_3909C80(a1, v10, v19, v37) )
  {
    return 1;
  }
  else
  {
    v38 = 259;
    v37[0] = "Expected End of Statement";
    v3 = sub_3909E20(a1, 9, v37);
    if ( !(_BYTE)v3 )
    {
      v11 = *(_QWORD *)(a1 + 320);
      v37[0] = v20;
      v38 = 261;
      v12 = sub_38BF510(v11, (__int64)v37);
      v13 = *(_QWORD *)(a1 + 320);
      v37[0] = v21;
      v14 = v12;
      v38 = 261;
      v15 = sub_38BF510(v13, (__int64)v37);
      (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(a1 + 328) + 640LL))(
        *(_QWORD *)(a1 + 328),
        (unsigned int)v16,
        (unsigned int)v17,
        (unsigned int)v18,
        v14,
        v15);
    }
  }
  return v3;
}
