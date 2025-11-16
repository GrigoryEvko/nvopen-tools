// Function: sub_BCA5C0
// Address: 0xbca5c0
//
__int64 __fastcall sub_BCA5C0(int *a1, __int64 *a2, char a3, char a4)
{
  const char *v6; // r15
  size_t v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rbx
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  __int64 *v31; // rsi
  __int64 v32; // r12
  __int64 v34; // rbx
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  __int64 v37; // rbx
  unsigned __int8 *v38; // rax
  __int64 v39; // rbx
  __int64 v40; // rax
  unsigned __int64 v41; // rdx
  __int64 v42; // [rsp+0h] [rbp-100h]
  __int64 v43; // [rsp+0h] [rbp-100h]
  __int64 v44; // [rsp+0h] [rbp-100h]
  __int64 v45; // [rsp+0h] [rbp-100h]
  __int64 v46; // [rsp+0h] [rbp-100h]
  __int64 v47; // [rsp+0h] [rbp-100h]
  __int64 v48; // [rsp+0h] [rbp-100h]
  double v50; // [rsp+8h] [rbp-F8h]
  __int64 v51; // [rsp+10h] [rbp-F0h] BYREF
  __int64 v52; // [rsp+18h] [rbp-E8h]
  char *s[4]; // [rsp+20h] [rbp-E0h]
  __int64 *v54; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v55; // [rsp+48h] [rbp-B8h]
  _BYTE v56[176]; // [rsp+50h] [rbp-B0h] BYREF

  s[0] = "InstrProf";
  s[1] = "CSInstrProf";
  s[2] = "SampleProfile";
  v55 = 0x1000000000LL;
  v6 = s[*a1];
  v54 = (__int64 *)v56;
  v51 = sub_B9B140(a2, "ProfileFormat", 0xDu);
  if ( v6 )
  {
    v7 = strlen(v6);
  }
  else
  {
    v6 = 0;
    v7 = 0;
  }
  v52 = sub_B9B140(a2, v6, v7);
  v8 = sub_B9C770(a2, &v51, (__int64 *)2, 0, 1);
  v9 = (unsigned int)v55;
  if ( (unsigned __int64)(unsigned int)v55 + 1 > HIDWORD(v55) )
  {
    v48 = v8;
    sub_C8D5F0(&v54, v56, (unsigned int)v55 + 1LL, 8);
    v9 = (unsigned int)v55;
    v8 = v48;
  }
  v54[v9] = v8;
  v10 = *((_QWORD *)a1 + 4);
  LODWORD(v55) = v55 + 1;
  v11 = sub_BC9630(a2, "TotalCount", v10);
  v12 = (unsigned int)v55;
  if ( (unsigned __int64)(unsigned int)v55 + 1 > HIDWORD(v55) )
  {
    v42 = v11;
    sub_C8D5F0(&v54, v56, (unsigned int)v55 + 1LL, 8);
    v12 = (unsigned int)v55;
    v11 = v42;
  }
  v54[v12] = v11;
  v13 = *((_QWORD *)a1 + 5);
  LODWORD(v55) = v55 + 1;
  v14 = sub_BC9630(a2, "MaxCount", v13);
  v15 = (unsigned int)v55;
  if ( (unsigned __int64)(unsigned int)v55 + 1 > HIDWORD(v55) )
  {
    v43 = v14;
    sub_C8D5F0(&v54, v56, (unsigned int)v55 + 1LL, 8);
    v15 = (unsigned int)v55;
    v14 = v43;
  }
  v54[v15] = v14;
  v16 = *((_QWORD *)a1 + 6);
  LODWORD(v55) = v55 + 1;
  v17 = sub_BC9630(a2, "MaxInternalCount", v16);
  v18 = (unsigned int)v55;
  if ( (unsigned __int64)(unsigned int)v55 + 1 > HIDWORD(v55) )
  {
    v44 = v17;
    sub_C8D5F0(&v54, v56, (unsigned int)v55 + 1LL, 8);
    v18 = (unsigned int)v55;
    v17 = v44;
  }
  v54[v18] = v17;
  v19 = *((_QWORD *)a1 + 7);
  LODWORD(v55) = v55 + 1;
  v20 = sub_BC9630(a2, "MaxFunctionCount", v19);
  v21 = (unsigned int)v55;
  if ( (unsigned __int64)(unsigned int)v55 + 1 > HIDWORD(v55) )
  {
    v45 = v20;
    sub_C8D5F0(&v54, v56, (unsigned int)v55 + 1LL, 8);
    v21 = (unsigned int)v55;
    v20 = v45;
  }
  v54[v21] = v20;
  v22 = (unsigned int)a1[16];
  LODWORD(v55) = v55 + 1;
  v23 = sub_BC9630(a2, "NumCounts", v22);
  v24 = (unsigned int)v55;
  if ( (unsigned __int64)(unsigned int)v55 + 1 > HIDWORD(v55) )
  {
    v46 = v23;
    sub_C8D5F0(&v54, v56, (unsigned int)v55 + 1LL, 8);
    v24 = (unsigned int)v55;
    v23 = v46;
  }
  v54[v24] = v23;
  v25 = (unsigned int)a1[17];
  LODWORD(v55) = v55 + 1;
  v26 = sub_BC9630(a2, "NumFunctions", v25);
  v27 = (unsigned int)v55;
  if ( (unsigned __int64)(unsigned int)v55 + 1 > HIDWORD(v55) )
  {
    v47 = v26;
    sub_C8D5F0(&v54, v56, (unsigned int)v55 + 1LL, 8);
    v27 = (unsigned int)v55;
    v26 = v47;
  }
  v54[v27] = v26;
  LODWORD(v55) = v55 + 1;
  if ( !a3 )
  {
    if ( !a4 )
      goto LABEL_19;
    goto LABEL_27;
  }
  v34 = sub_BC9630(a2, "IsPartialProfile", *((unsigned __int8 *)a1 + 72));
  v35 = (unsigned int)v55;
  v36 = (unsigned int)v55 + 1LL;
  if ( v36 > HIDWORD(v55) )
  {
    sub_C8D5F0(&v54, v56, v36, 8);
    v35 = (unsigned int)v55;
  }
  v54[v35] = v34;
  LODWORD(v55) = v55 + 1;
  if ( a4 )
  {
LABEL_27:
    v50 = *((double *)a1 + 10);
    v37 = sub_BCB170(a2);
    v51 = sub_B9B140(a2, "PartialProfileRatio", 0x13u);
    v38 = sub_AD8DD0(v37, v50);
    v52 = (__int64)sub_B98A20((__int64)v38, (__int64)"PartialProfileRatio");
    v39 = sub_B9C770(a2, &v51, (__int64 *)2, 0, 1);
    v40 = (unsigned int)v55;
    v41 = (unsigned int)v55 + 1LL;
    if ( v41 > HIDWORD(v55) )
    {
      sub_C8D5F0(&v54, v56, v41, 8);
      v40 = (unsigned int)v55;
    }
    v54[v40] = v39;
    LODWORD(v55) = v55 + 1;
  }
LABEL_19:
  v28 = sub_BCA410((__int64)a1, a2);
  v29 = (unsigned int)v55;
  v30 = (unsigned int)v55 + 1LL;
  if ( v30 > HIDWORD(v55) )
  {
    sub_C8D5F0(&v54, v56, v30, 8);
    v29 = (unsigned int)v55;
  }
  v54[v29] = v28;
  v31 = v54;
  LODWORD(v55) = v55 + 1;
  v32 = sub_B9C770(a2, v54, (__int64 *)(unsigned int)v55, 0, 1);
  if ( v54 != (__int64 *)v56 )
    _libc_free(v54, v31);
  return v32;
}
