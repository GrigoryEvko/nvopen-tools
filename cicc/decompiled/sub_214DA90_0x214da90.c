// Function: sub_214DA90
// Address: 0x214da90
//
__int64 __fastcall sub_214DA90(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // r14
  bool v6; // bl
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // rax
  _WORD *v19; // rdx
  __int64 v20; // rdi
  __int64 v21; // rsi
  __int64 v22; // rax
  _WORD *v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 result; // rax
  const char *v28; // rax
  __int64 v29; // rdx
  const char *v30; // rsi
  const char *v31; // rdi
  int v32; // edx
  int v33; // ecx
  int v34; // ebx
  __int64 v35; // rbx
  __int64 v36; // rax
  __int64 v37; // r10
  __int64 v38; // rbx
  __int64 v39; // r9
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  char v48; // r15
  char v49; // bl
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  const char *v54; // rax
  __int64 v55; // rdx
  const char *v56; // rsi
  const char *v57; // rdi
  int v58; // edx
  int v59; // ecx
  int v60; // ebx
  __int64 v61; // r15
  __int64 v62; // rax
  __int64 v63; // r14
  __int64 v64; // r15
  __int64 v65; // rbx
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // [rsp+0h] [rbp-B0h]
  int v81; // [rsp+18h] [rbp-98h]
  __int64 v82; // [rsp+18h] [rbp-98h]
  unsigned int v83; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v84; // [rsp+24h] [rbp-8Ch] BYREF
  unsigned int v85; // [rsp+28h] [rbp-88h] BYREF
  unsigned int v86; // [rsp+2Ch] [rbp-84h] BYREF
  __int64 v87; // [rsp+30h] [rbp-80h] BYREF
  char *v88; // [rsp+38h] [rbp-78h] BYREF
  unsigned int v89; // [rsp+40h] [rbp-70h] BYREF
  char v90; // [rsp+44h] [rbp-6Ch]
  unsigned int v91; // [rsp+48h] [rbp-68h] BYREF
  char v92; // [rsp+4Ch] [rbp-64h]
  unsigned int v93; // [rsp+50h] [rbp-60h] BYREF
  char v94; // [rsp+54h] [rbp-5Ch]
  unsigned int v95; // [rsp+58h] [rbp-58h] BYREF
  char v96; // [rsp+5Ch] [rbp-54h]
  __int64 v97; // [rsp+60h] [rbp-50h] BYREF
  char *endptr; // [rsp+68h] [rbp-48h] BYREF
  __int64 v99; // [rsp+74h] [rbp-3Ch] BYREF
  int v100; // [rsp+7Ch] [rbp-34h]

  v5 = (_QWORD *)(a2 + 112);
  sub_1C2EDB0((__int64)&v89, a2);
  sub_1C2EE00((__int64)&v91, a2);
  sub_1C2EE50((__int64)&v93, a2);
  v6 = sub_15602E0((_QWORD *)(a2 + 112), "nvvm.reqntid", 0xCu);
  if ( sub_15602E0((_QWORD *)(a2 + 112), "nvvm.blocksareclusters", 0x16u) )
  {
    sub_1263B40(a3, ".blocksareclusters\n");
    if ( !v6 )
      sub_16BD130("blocksareclusters requires reqntid", 1u);
LABEL_32:
    v97 = sub_1560340(v5, -1, "nvvm.reqntid", 0xCu);
    v28 = (const char *)sub_155D8B0(&v97);
    v30 = &v28[v29];
    v31 = v28;
    if ( &v28[v29] == v28 )
    {
      v100 = 1;
      v99 = 0x100000001LL;
      v81 = 0;
    }
    else
    {
      v32 = 0;
      do
      {
        v33 = *v28++ == 44;
        v32 += v33;
      }
      while ( v30 != v28 );
      v34 = 2;
      v100 = 1;
      if ( v32 <= 2 )
        v34 = v32;
      v99 = 0x100000001LL;
      v81 = v34;
      if ( v32 < 0 )
      {
        v39 = 1;
        v38 = 1;
        v37 = 1;
LABEL_43:
        v82 = v39;
        v79 = v37;
        v40 = sub_1263B40(a3, ".reqntid ");
        v41 = sub_16E7AB0(v40, v79);
        v42 = sub_1263B40(v41, ", ");
        v43 = sub_16E7AB0(v42, v82);
        v44 = sub_1263B40(v43, ", ");
        v45 = sub_16E7AB0(v44, v38);
        sub_1263B40(v45, "\n");
        goto LABEL_13;
      }
    }
    v35 = 0;
    do
    {
      endptr = 0;
      v36 = strtol(v31, &endptr, 10);
      if ( !v36 )
LABEL_40:
        sub_16BD130("Expects a number as a value.", 1u);
      *((_DWORD *)&v99 + v35++) = v36;
      v31 = endptr + 1;
    }
    while ( v81 >= (int)v35 );
    v37 = (int)v99;
    v38 = v100;
    v39 = SHIDWORD(v99);
    goto LABEL_43;
  }
  if ( v6 )
    goto LABEL_32;
  if ( v90 || v92 || v94 )
  {
    v7 = 1;
    v8 = sub_1263B40(a3, ".reqntid ");
    if ( v90 )
      v7 = v89;
    v9 = sub_16E7A90(v8, v7);
    v10 = 1;
    v11 = sub_1263B40(v9, ", ");
    if ( v92 )
      v10 = v91;
    v12 = sub_16E7A90(v11, v10);
    v13 = 1;
    v14 = sub_1263B40(v12, ", ");
    if ( v94 )
      v13 = v93;
    v15 = sub_16E7A90(v14, v13);
    sub_1263B40(v15, "\n");
  }
LABEL_13:
  sub_1C2EC00((__int64)&v95, a2);
  sub_1C2EC50((__int64)&v97, a2);
  sub_1C2ECA0((__int64)&endptr, a2);
  if ( v96 || BYTE4(v97) || BYTE4(endptr) )
  {
    v16 = 1;
    v17 = sub_1263B40(a3, ".maxntid ");
    if ( v96 )
      v16 = v95;
    v18 = sub_16E7A90(v17, v16);
    v19 = *(_WORD **)(v18 + 24);
    v20 = v18;
    if ( *(_QWORD *)(v18 + 16) - (_QWORD)v19 <= 1u )
    {
      v21 = 1;
      v20 = sub_16E7EE0(v18, ", ", 2u);
      if ( !BYTE4(v97) )
      {
LABEL_19:
        v22 = sub_16E7A90(v20, v21);
        v23 = *(_WORD **)(v22 + 24);
        v24 = v22;
        if ( *(_QWORD *)(v22 + 16) - (_QWORD)v23 <= 1u )
        {
          v25 = 1;
          v24 = sub_16E7EE0(v22, ", ", 2u);
          if ( !BYTE4(endptr) )
            goto LABEL_21;
        }
        else
        {
          v25 = 1;
          *v23 = 8236;
          *(_QWORD *)(v22 + 24) += 2LL;
          if ( !BYTE4(endptr) )
          {
LABEL_21:
            v26 = sub_16E7A90(v24, v25);
            sub_1263B40(v26, "\n");
            goto LABEL_22;
          }
        }
        v25 = (unsigned int)endptr;
        goto LABEL_21;
      }
    }
    else
    {
      *v19 = 8236;
      v21 = 1;
      *(_QWORD *)(v18 + 24) += 2LL;
      if ( !BYTE4(v97) )
        goto LABEL_19;
    }
    v21 = (unsigned int)v97;
    goto LABEL_19;
  }
LABEL_22:
  if ( (unsigned __int8)sub_1C2EF70(a2, &v83) )
  {
    v46 = sub_1263B40(a3, ".minnctapersm ");
    v47 = sub_16E7A90(v46, v83);
    sub_1263B40(v47, "\n");
  }
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 232) + 1212LL) <= 0x59u )
  {
LABEL_25:
    result = sub_1C2EF90(a2, &v99);
    if ( !(_BYTE)result )
      return result;
LABEL_53:
    v52 = sub_1263B40(a3, ".maxnreg ");
    v53 = sub_16E7A90(v52, (unsigned int)v99);
    return sub_1263B40(v53, "\n");
  }
  v48 = sub_1C2EEF0(a2, &v84);
  if ( !v48 )
    v84 = 1;
  v49 = sub_1C2EF10(a2, &v85);
  if ( !v49 )
  {
    v85 = 1;
    v49 = v48;
  }
  if ( !(unsigned __int8)sub_1C2EF30(a2, &v86) )
  {
    v86 = 1;
    if ( !sub_15602E0(v5, "nvvm.cluster_dim", 0x10u) )
    {
      if ( !v49 )
        goto LABEL_51;
      goto LABEL_70;
    }
LABEL_59:
    v87 = sub_1560340(v5, -1, "nvvm.cluster_dim", 0x10u);
    v54 = (const char *)sub_155D8B0(&v87);
    v56 = &v54[v55];
    v57 = v54;
    if ( &v54[v55] == v54 )
    {
      v100 = 1;
      v60 = 0;
      v99 = 0x100000001LL;
    }
    else
    {
      v58 = 0;
      do
      {
        v59 = *v54++ == 44;
        v58 += v59;
      }
      while ( v56 != v54 );
      v60 = 2;
      v100 = 1;
      v99 = 0x100000001LL;
      if ( v58 <= 2 )
        v60 = v58;
      if ( v58 < 0 )
      {
        v65 = 1;
        v64 = 1;
        v63 = 1;
LABEL_69:
        v66 = sub_1263B40(a3, ".reqnctapercluster ");
        v67 = sub_16E7AB0(v66, v65);
        v68 = sub_1263B40(v67, ", ");
        v69 = sub_16E7AB0(v68, v64);
        v70 = sub_1263B40(v69, ", ");
        v71 = sub_16E7AB0(v70, v63);
        sub_1263B40(v71, "\n");
        goto LABEL_51;
      }
    }
    v61 = 0;
    do
    {
      v88 = 0;
      v62 = strtol(v57, &v88, 10);
      if ( !v62 )
        goto LABEL_40;
      *((_DWORD *)&v99 + v61++) = v62;
      v57 = v88 + 1;
    }
    while ( v60 >= (int)v61 );
    v63 = v100;
    v64 = SHIDWORD(v99);
    v65 = (int)v99;
    goto LABEL_69;
  }
  if ( sub_15602E0(v5, "nvvm.cluster_dim", 0x10u) )
    goto LABEL_59;
LABEL_70:
  v72 = sub_1263B40(a3, ".explicitcluster");
  sub_1263B40(v72, "\n");
  if ( v84 )
  {
    v73 = sub_1263B40(a3, ".reqnctapercluster ");
    v74 = sub_16E7A90(v73, v84);
    v75 = sub_1263B40(v74, ", ");
    v76 = sub_16E7A90(v75, v85);
    v77 = sub_1263B40(v76, ", ");
    v78 = sub_16E7A90(v77, v86);
    sub_1263B40(v78, "\n");
  }
LABEL_51:
  if ( !(unsigned __int8)sub_1C2EF50(a2, &v99) )
    goto LABEL_25;
  v50 = sub_1263B40(a3, ".maxclusterrank ");
  v51 = sub_16E7A90(v50, (unsigned int)v99);
  sub_1263B40(v51, "\n");
  result = sub_1C2EF90(a2, &v99);
  if ( (_BYTE)result )
    goto LABEL_53;
  return result;
}
