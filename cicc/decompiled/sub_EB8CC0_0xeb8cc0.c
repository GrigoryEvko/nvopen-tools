// Function: sub_EB8CC0
// Address: 0xeb8cc0
//
__int64 __fastcall sub_EB8CC0(__int64 a1)
{
  __int64 v2; // rax
  unsigned int v3; // r12d
  unsigned __int8 v5; // al
  unsigned __int8 v6; // al
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // [rsp+0h] [rbp-180h] BYREF
  __int64 v13; // [rsp+8h] [rbp-178h] BYREF
  unsigned __int64 v14; // [rsp+10h] [rbp-170h] BYREF
  __int64 v15; // [rsp+18h] [rbp-168h] BYREF
  const char *v16; // [rsp+20h] [rbp-160h] BYREF
  __int64 v17; // [rsp+28h] [rbp-158h]
  const char *v18; // [rsp+30h] [rbp-150h] BYREF
  __int64 v19; // [rsp+38h] [rbp-148h]
  const char *v20; // [rsp+40h] [rbp-140h] BYREF
  char v21; // [rsp+60h] [rbp-120h]
  char v22; // [rsp+61h] [rbp-11Fh]
  const char *v23; // [rsp+70h] [rbp-110h] BYREF
  char v24; // [rsp+90h] [rbp-F0h]
  char v25; // [rsp+91h] [rbp-EFh]
  const char *v26; // [rsp+A0h] [rbp-E0h] BYREF
  char v27; // [rsp+C0h] [rbp-C0h]
  char v28; // [rsp+C1h] [rbp-BFh]
  const char *v29; // [rsp+D0h] [rbp-B0h] BYREF
  char v30; // [rsp+F0h] [rbp-90h]
  char v31; // [rsp+F1h] [rbp-8Fh]
  const char *v32; // [rsp+100h] [rbp-80h] BYREF
  char v33; // [rsp+120h] [rbp-60h]
  char v34; // [rsp+121h] [rbp-5Fh]
  const char *v35; // [rsp+130h] [rbp-50h] BYREF
  __int64 v36; // [rsp+138h] [rbp-48h]
  __int16 v37; // [rsp+150h] [rbp-30h]

  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v2 = sub_ECD7B0(a1);
  v15 = sub_ECD6A0(v2);
  if ( (unsigned __int8)sub_EA2660(a1, &v12) )
    return 1;
  if ( (unsigned __int8)sub_ECD7C0(a1, &v15) )
    return 1;
  v22 = 1;
  v20 = "expected SourceField";
  v21 = 3;
  if ( (unsigned __int8)sub_ECE130(a1, &v13, &v20) )
    return 1;
  v25 = 1;
  v23 = "File id less than zero";
  v24 = 3;
  if ( (unsigned __int8)sub_ECE070(a1, v13 <= 0, v15, &v23) )
    return 1;
  if ( (unsigned __int8)sub_ECD7C0(a1, &v15) )
    return 1;
  v28 = 1;
  v26 = "expected SourceLineNum";
  v27 = 3;
  if ( (unsigned __int8)sub_ECE130(a1, &v14, &v26) )
    return 1;
  v31 = 1;
  v29 = "Line number less than zero";
  v30 = 3;
  if ( (unsigned __int8)sub_ECE070(a1, v14 >> 63, v15, &v29) )
    return 1;
  if ( (unsigned __int8)sub_ECD7C0(a1, &v15) )
    return 1;
  v34 = 1;
  v32 = "expected identifier";
  v33 = 3;
  v5 = sub_EB61F0(a1, (__int64 *)&v16);
  if ( (unsigned __int8)sub_ECE070(a1, v5, v15, &v32) )
    return 1;
  if ( (unsigned __int8)sub_ECD7C0(a1, &v15) )
    return 1;
  v35 = "expected identifier";
  v37 = 259;
  v6 = sub_EB61F0(a1, (__int64 *)&v18);
  if ( (unsigned __int8)sub_ECE070(a1, v6, v15, &v35) )
  {
    return 1;
  }
  else
  {
    v3 = sub_ECE000(a1);
    if ( !(_BYTE)v3 )
    {
      v7 = *(_QWORD *)(a1 + 224);
      v37 = 261;
      v35 = v16;
      v36 = v17;
      v8 = sub_E6C460(v7, &v35);
      v9 = *(_QWORD *)(a1 + 224);
      v37 = 261;
      v10 = v8;
      v35 = v18;
      v36 = v19;
      v11 = sub_E6C460(v9, &v35);
      (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(a1 + 232) + 752LL))(
        *(_QWORD *)(a1 + 232),
        (unsigned int)v12,
        (unsigned int)v13,
        (unsigned int)v14,
        v10,
        v11);
    }
  }
  return v3;
}
