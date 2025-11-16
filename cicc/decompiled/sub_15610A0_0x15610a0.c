// Function: sub_15610A0
// Address: 0x15610a0
//
_DWORD *__fastcall sub_15610A0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r13
  _DWORD *v3; // rax
  __int64 v4; // rdx
  _DWORD *v5; // rax
  __int64 v6; // rdx
  _DWORD *v7; // rax
  __int64 v8; // rdx
  _DWORD *v9; // rax
  __int64 v10; // rdx
  _QWORD *v11; // rax
  _QWORD *v12; // rax
  _DWORD *v13; // rax
  __int64 v14; // rdx
  _DWORD *v15; // rax
  __int64 v16; // rdx
  _DWORD *v17; // rax
  __int64 v18; // rdx
  _DWORD *result; // rax
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 *v22; // rax
  _QWORD *v23; // rax
  _DWORD *v24; // rax
  __int64 v25; // rdx
  __int64 *v26; // rax
  _QWORD *v27; // rax
  _DWORD *v28; // rax
  __int64 v29; // rdx
  __int64 *v30; // rax
  _QWORD *v31; // rax
  _DWORD *v32; // rax
  __int64 v33; // rdx
  __int64 *v34; // rax
  _QWORD *v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rdx
  char v39; // al
  __int64 v40; // r9
  __int64 v41; // rax
  __int64 v42; // rdx
  bool v43; // zf
  unsigned __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rdx
  char v48; // al
  __int64 v49; // r9
  __int64 v50; // rax
  __int64 v51; // rdx
  unsigned __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 *v55; // rax
  _QWORD *v56; // rax
  __int64 *v57; // rax
  _QWORD *v58; // rax
  unsigned __int64 v59; // [rsp+18h] [rbp-A8h]
  unsigned __int64 v60; // [rsp+18h] [rbp-A8h]
  __int64 v61; // [rsp+28h] [rbp-98h] BYREF
  __int64 v62[2]; // [rsp+30h] [rbp-90h] BYREF
  int v63; // [rsp+40h] [rbp-80h] BYREF
  _QWORD *v64; // [rsp+48h] [rbp-78h]
  int *v65; // [rsp+50h] [rbp-70h]
  int *v66; // [rsp+58h] [rbp-68h]
  __int64 v67; // [rsp+60h] [rbp-60h]
  __int64 v68; // [rsp+68h] [rbp-58h]
  __int64 v69; // [rsp+70h] [rbp-50h]
  __int64 v70; // [rsp+78h] [rbp-48h]
  __int64 v71; // [rsp+80h] [rbp-40h]
  __int64 v72; // [rsp+88h] [rbp-38h]

  v2 = (_QWORD *)(a1 + 112);
  if ( !(unsigned __int8)sub_1560180(a1 + 112, 25) && (unsigned __int8)sub_1560180(a2 + 112, 25) )
    sub_15E0D50(a1, 0xFFFFFFFFLL, 25);
  v62[0] = sub_1560340(v2, -1, "no-jump-tables", 0xEu);
  v3 = (_DWORD *)sub_155D8B0(v62);
  if ( v4 != 4 || *v3 != 1702195828 )
  {
    v62[0] = sub_1560340((_QWORD *)(a2 + 112), -1, "no-jump-tables", 0xEu);
    v5 = (_DWORD *)sub_155D8B0(v62);
    if ( v6 == 4 && *v5 == 1702195828 )
    {
      v55 = (__int64 *)sub_15E0530(a1);
      v56 = sub_155D020(v55, "no-jump-tables", 0xEu, "true", 4u);
      sub_15E0DA0(a1, 0xFFFFFFFFLL, v56);
    }
  }
  v62[0] = sub_1560340(v2, -1, "profile-sample-accurate", 0x17u);
  v7 = (_DWORD *)sub_155D8B0(v62);
  if ( v8 != 4 || *v7 != 1702195828 )
  {
    v62[0] = sub_1560340((_QWORD *)(a2 + 112), -1, "profile-sample-accurate", 0x17u);
    v9 = (_DWORD *)sub_155D8B0(v62);
    if ( v10 == 4 && *v9 == 1702195828 )
    {
      v57 = (__int64 *)sub_15E0530(a1);
      v58 = sub_155D020(v57, "profile-sample-accurate", 0x17u, "true", 4u);
      sub_15E0DA0(a1, 0xFFFFFFFFLL, v58);
    }
  }
  v62[0] = 0;
  v63 = 0;
  v64 = 0;
  v65 = &v63;
  v66 = &v63;
  v67 = 0;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  v72 = 0;
  v11 = sub_15606E0(v62, 49);
  v12 = sub_15606E0(v11, 51);
  sub_15606E0(v12, 50);
  if ( (unsigned __int8)sub_1560180(a2 + 112, 50) )
  {
    sub_15E0EF0(a1, 0xFFFFFFFFLL, v62);
    sub_15E0D50(a1, 0xFFFFFFFFLL, 50);
  }
  else if ( (unsigned __int8)sub_1560180(a2 + 112, 51) && !(unsigned __int8)sub_1560180((__int64)v2, 50) )
  {
    sub_15E0EF0(a1, 0xFFFFFFFFLL, v62);
    sub_15E0D50(a1, 0xFFFFFFFFLL, 51);
  }
  else if ( (unsigned __int8)sub_1560180(a2 + 112, 49)
         && !(unsigned __int8)sub_1560180((__int64)v2, 50)
         && !(unsigned __int8)sub_1560180((__int64)v2, 51) )
  {
    sub_15E0D50(a1, 0xFFFFFFFFLL, 49);
  }
  sub_155CC10(v64);
  if ( !sub_15602E0(v2, "probe-stack", 0xBu) && sub_15602E0((_QWORD *)(a2 + 112), "probe-stack", 0xBu) )
  {
    v54 = sub_1560340((_QWORD *)(a2 + 112), -1, "probe-stack", 0xBu);
    sub_15E0DA0(a1, 0xFFFFFFFFLL, v54);
  }
  if ( sub_15602E0((_QWORD *)(a2 + 112), "stack-probe-size", 0x10u) )
  {
    v61 = sub_1560340((_QWORD *)(a2 + 112), -1, "stack-probe-size", 0x10u);
    v46 = sub_155D8B0(&v61);
    v48 = sub_16D2B80(v46, v47, 0, v62);
    v49 = 0;
    if ( !v48 )
      v49 = v62[0];
    v60 = v49;
    if ( !sub_15602E0(v2, "stack-probe-size", 0x10u) )
      goto LABEL_56;
    v61 = sub_1560340(v2, -1, "stack-probe-size", 0x10u);
    v50 = sub_155D8B0(&v61);
    v43 = (unsigned __int8)sub_16D2B80(v50, v51, 0, v62) == 0;
    v52 = 0;
    if ( v43 )
      v52 = v62[0];
    if ( v60 < v52 )
    {
LABEL_56:
      v53 = sub_1560340((_QWORD *)(a2 + 112), -1, "stack-probe-size", 0x10u);
      sub_15E0DA0(a1, 0xFFFFFFFFLL, v53);
    }
  }
  if ( sub_15602E0((_QWORD *)(a2 + 112), "min-legal-vector-width", 0x16u) )
  {
    v61 = sub_1560340((_QWORD *)(a2 + 112), -1, "min-legal-vector-width", 0x16u);
    v37 = sub_155D8B0(&v61);
    v39 = sub_16D2B80(v37, v38, 0, v62);
    v40 = 0;
    if ( !v39 )
      v40 = v62[0];
    v59 = v40;
    if ( !sub_15602E0(v2, "min-legal-vector-width", 0x16u) )
      goto LABEL_49;
    v61 = sub_1560340(v2, -1, "min-legal-vector-width", 0x16u);
    v41 = sub_155D8B0(&v61);
    v43 = (unsigned __int8)sub_16D2B80(v41, v42, 0, v62) == 0;
    v44 = 0;
    if ( v43 )
      v44 = v62[0];
    if ( v59 > v44 )
    {
LABEL_49:
      v45 = sub_1560340((_QWORD *)(a2 + 112), -1, "min-legal-vector-width", 0x16u);
      sub_15E0DA0(a1, 0xFFFFFFFFLL, v45);
    }
  }
  if ( (unsigned __int8)sub_15E4640(a2) && !(unsigned __int8)sub_15E4640(a1) )
  {
    v36 = sub_1560340((_QWORD *)(a2 + 112), -1, "null-pointer-is-valid", 0x15u);
    sub_15E0DA0(a1, 0xFFFFFFFFLL, v36);
  }
  v62[0] = sub_1560340(v2, -1, "less-precise-fpmad", 0x12u);
  v13 = (_DWORD *)sub_155D8B0(v62);
  if ( v14 == 4 && *v13 == 1702195828 )
  {
    v62[0] = sub_1560340((_QWORD *)(a2 + 112), -1, "less-precise-fpmad", 0x12u);
    v32 = (_DWORD *)sub_155D8B0(v62);
    if ( v33 != 4 || *v32 != 1702195828 )
    {
      v34 = (__int64 *)sub_15E0530(a1);
      v35 = sub_155D020(v34, "less-precise-fpmad", 0x12u, "false", 5u);
      sub_15E0DA0(a1, 0xFFFFFFFFLL, v35);
    }
  }
  v62[0] = sub_1560340(v2, -1, "no-infs-fp-math", 0xFu);
  v15 = (_DWORD *)sub_155D8B0(v62);
  if ( v16 == 4 && *v15 == 1702195828 )
  {
    v62[0] = sub_1560340((_QWORD *)(a2 + 112), -1, "no-infs-fp-math", 0xFu);
    v28 = (_DWORD *)sub_155D8B0(v62);
    if ( v29 != 4 || *v28 != 1702195828 )
    {
      v30 = (__int64 *)sub_15E0530(a1);
      v31 = sub_155D020(v30, "no-infs-fp-math", 0xFu, "false", 5u);
      sub_15E0DA0(a1, 0xFFFFFFFFLL, v31);
    }
  }
  v62[0] = sub_1560340(v2, -1, "no-nans-fp-math", 0xFu);
  v17 = (_DWORD *)sub_155D8B0(v62);
  if ( v18 == 4 && *v17 == 1702195828 )
  {
    v62[0] = sub_1560340((_QWORD *)(a2 + 112), -1, "no-nans-fp-math", 0xFu);
    v24 = (_DWORD *)sub_155D8B0(v62);
    if ( v25 != 4 || *v24 != 1702195828 )
    {
      v26 = (__int64 *)sub_15E0530(a1);
      v27 = sub_155D020(v26, "no-nans-fp-math", 0xFu, "false", 5u);
      sub_15E0DA0(a1, 0xFFFFFFFFLL, v27);
    }
  }
  v62[0] = sub_1560340(v2, -1, "unsafe-fp-math", 0xEu);
  result = (_DWORD *)sub_155D8B0(v62);
  if ( v20 == 4 && *result == 1702195828 )
  {
    v62[0] = sub_1560340((_QWORD *)(a2 + 112), -1, "unsafe-fp-math", 0xEu);
    result = (_DWORD *)sub_155D8B0(v62);
    if ( v21 != 4 || *result != 1702195828 )
    {
      v22 = (__int64 *)sub_15E0530(a1);
      v23 = sub_155D020(v22, "unsafe-fp-math", 0xEu, "false", 5u);
      return (_DWORD *)sub_15E0DA0(a1, 0xFFFFFFFFLL, v23);
    }
  }
  return result;
}
