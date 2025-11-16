// Function: sub_148AAB0
// Address: 0x148aab0
//
__int64 __fastcall sub_148AAB0(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v8; // rax
  unsigned __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r9
  __int64 v14; // r9
  __int64 v15; // rcx
  __int64 v16; // rdi
  unsigned int v17; // eax
  unsigned int v18; // eax
  __int64 v19; // r13
  __int64 v20; // r15
  __int64 v21; // rbx
  __int64 v22; // r14
  __int64 result; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned __int64 v27; // rbx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdx
  char v32; // al
  __int64 v33; // rdi
  unsigned int v34; // eax
  unsigned int v35; // eax
  __int64 v36; // rbx
  __int64 v37; // r14
  __int64 *v38; // rax
  __int64 v39; // rsi
  __int64 v40; // rax
  __int64 v41; // rax
  int v42; // eax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 *v45; // rax
  __int64 v46; // [rsp+0h] [rbp-80h] BYREF
  __int64 v47; // [rsp+8h] [rbp-78h] BYREF
  __int64 v48; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v49; // [rsp+18h] [rbp-68h] BYREF
  unsigned int v50; // [rsp+1Ch] [rbp-64h] BYREF
  __int64 v51; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v52; // [rsp+28h] [rbp-58h]
  __int64 v53; // [rsp+30h] [rbp-50h] BYREF
  int v54; // [rsp+38h] [rbp-48h]
  __int64 v55; // [rsp+40h] [rbp-40h] BYREF
  int v56; // [rsp+48h] [rbp-38h]

  v48 = a3;
  v47 = a4;
  v49 = a5;
  v46 = a6;
  v50 = a2;
  v8 = sub_1456040(a3);
  v9 = sub_1456C90(a1, v8);
  v10 = sub_1456040(v46);
  if ( v9 >= sub_1456C90(a1, v10) )
  {
    v26 = sub_1456040(v48);
    v27 = sub_1456C90(a1, v26);
    v28 = sub_1456040(v46);
    if ( v27 > sub_1456C90(a1, v28) )
    {
      if ( (unsigned __int8)sub_15FF7F0(v49) )
      {
        v29 = sub_1456040(v48);
        v46 = sub_147B0D0(a1, v46, v29, 0);
        v30 = sub_1456040(v48);
        a7 = sub_147B0D0(a1, a7, v30, 0);
      }
      else
      {
        v43 = sub_1456040(v48);
        v46 = sub_14747F0(a1, v46, v43, 0);
        v44 = sub_1456040(v48);
        a7 = sub_14747F0(a1, a7, v44, 0);
      }
    }
  }
  else if ( (unsigned __int8)sub_15FF7F0(v50) )
  {
    v24 = sub_1456040(v46);
    v48 = sub_147B0D0(a1, v48, v24, 0);
    v25 = sub_1456040(v46);
    v47 = sub_147B0D0(a1, v47, v25, 0);
  }
  else
  {
    v11 = sub_1456040(v46);
    v48 = sub_14747F0(a1, v48, v11, 0);
    v12 = sub_1456040(v46);
    v47 = sub_14747F0(a1, v47, v12, 0);
  }
  if ( (unsigned __int8)sub_147DF40(a1, &v50, &v48, &v47, 0, v13) && v48 == v47 )
    return sub_15FF820(v50);
  if ( (unsigned __int8)sub_147DF40(a1, &v49, &v46, &a7, 0, v14) && v46 == a7 )
    return sub_15FF850(v49);
  v15 = v48;
  if ( v48 != a7 && v46 != v47 )
  {
    v16 = v49;
    v17 = v50;
    goto LABEL_11;
  }
  if ( *(_WORD *)(v47 + 24) )
  {
    v48 = v47;
    v47 = v15;
    v17 = sub_15FF5D0(v50);
    v16 = v49;
    v50 = v17;
LABEL_11:
    if ( (_DWORD)v16 != v17 )
      goto LABEL_12;
    return sub_148BC60(a1, (unsigned int)v16, v48, v47, v46, a7);
  }
  v31 = v46;
  v46 = a7;
  a7 = v31;
  v49 = sub_15FF5D0(v49);
  v16 = v49;
  if ( v49 == v50 )
    return sub_148BC60(a1, (unsigned int)v16, v48, v47, v46, a7);
LABEL_12:
  v18 = sub_15FF5D0(v16);
  if ( v18 == v50 )
  {
    v19 = v47;
    v20 = a7;
    v21 = v46;
    v22 = v48;
    if ( !*(_WORD *)(v47 + 24) )
      return sub_148BC60(a1, v18, v48, v47, a7, v46);
    v35 = sub_15FF5D0(v18);
    return sub_148BC60(a1, v35, v19, v22, v21, v20);
  }
  v32 = sub_15FF7E0(v49);
  v33 = v50;
  if ( v32 )
  {
    v42 = sub_15FF7D0(v49);
    v33 = v50;
    if ( v42 == v50 )
    {
      if ( (unsigned __int8)sub_1477BC0(a1, v46) && (unsigned __int8)sub_1477BC0(a1, a7) )
        return sub_148BC60(a1, v50, v48, v47, v46, a7);
      v33 = v50;
    }
  }
  v34 = v49;
  if ( v49 == 33 )
  {
    v36 = v46;
    v37 = a7;
    if ( *(_WORD *)(v46 + 24) )
    {
      if ( *(_WORD *)(a7 + 24) )
        goto LABEL_25;
      v36 = a7;
      v37 = v46;
    }
    if ( (unsigned __int8)sub_15FF7F0(v33) )
    {
      v45 = sub_1477920(a1, v37, 1u);
      sub_158ACE0(&v51, v45);
    }
    else
    {
      v38 = sub_1477920(a1, v37, 0);
      sub_158AAD0(&v51, v38);
    }
    v39 = *(_QWORD *)(v36 + 32);
    if ( v52 <= 0x40 )
    {
      if ( v51 != *(_QWORD *)(v39 + 24) )
        goto LABEL_43;
    }
    else if ( !(unsigned __int8)sub_16A5220(&v51, v39 + 24) )
    {
LABEL_43:
      sub_135E100(&v51);
      v34 = v49;
      v33 = v50;
      goto LABEL_24;
    }
    sub_13A38D0((__int64)&v55, (__int64)&v51);
    sub_16A7490(&v55, 1);
    v54 = v56;
    v53 = v55;
    if ( v50 != 38 )
    {
      if ( v50 > 0x26 )
      {
        if ( v50 != 39 )
          goto LABEL_42;
        goto LABEL_40;
      }
      if ( v50 != 34 )
      {
        if ( v50 != 35 )
          goto LABEL_42;
LABEL_40:
        v40 = sub_145CF40(a1, (__int64)&v53);
        if ( (unsigned __int8)sub_148BC60(a1, v50, v48, v47, v37, v40) )
          goto LABEL_62;
      }
    }
    v41 = sub_145CF40(a1, (__int64)&v51);
    if ( !(unsigned __int8)sub_148BC60(a1, v50, v48, v47, v37, v41) )
    {
LABEL_42:
      sub_135E100(&v53);
      goto LABEL_43;
    }
LABEL_62:
    sub_135E100(&v53);
    sub_135E100(&v51);
    return 1;
  }
LABEL_24:
  if ( v34 == 32 )
  {
    if ( (unsigned __int8)sub_15FF820(v33) )
    {
      result = sub_148BC60(a1, v50, v48, v47, v46, a7);
      if ( (_BYTE)result )
        return result;
    }
    LODWORD(v33) = v50;
  }
LABEL_25:
  if ( (_DWORD)v33 != 33 || (unsigned __int8)sub_15FF820(v49) )
    return 0;
  else
    return sub_148BC60(a1, v49, v48, v47, v46, a7);
}
