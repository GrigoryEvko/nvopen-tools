// Function: sub_2252E40
// Address: 0x2252e40
//
__int64 __fastcall sub_2252E40(int a1, int a2, __int64 a3, _BYTE *a4, __int64 a5)
{
  __int64 v5; // rbp
  __int64 v6; // r15
  int v7; // r13d
  __int64 result; // rax
  char *v9; // r12
  __int64 v10; // rax
  unsigned __int64 v11; // rbx
  char v12; // bp
  char *v13; // rax
  char *v14; // rax
  char v15; // bp
  char *v16; // r12
  char *v17; // rax
  char *v18; // rax
  char v19; // bp
  char *v20; // r12
  char *v21; // rsi
  char *v22; // rax
  __int64 v23; // rdi
  int v24; // ecx
  char v25; // dl
  unsigned __int64 v26; // rax
  __int64 v27; // r8
  _QWORD *v28; // rax
  __int64 **v29; // rax
  char v30; // bl
  __int64 v31; // r13
  __int64 v32; // r12
  char *v33; // rbp
  __int64 v34; // r15
  unsigned int v35; // ecx
  char v36; // dl
  unsigned __int64 v37; // rax
  char *v38; // rdx
  __int64 v39; // r14
  unsigned int v40; // ecx
  char v41; // si
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rdi
  __int64 v44; // rcx
  unsigned __int8 *v45; // rdx
  __int64 v46; // rsi
  __int64 v47; // rax
  _QWORD *v48; // rax
  bool v50; // [rsp+Bh] [rbp-CDh]
  char v51; // [rsp+Ch] [rbp-CCh]
  unsigned __int64 v52; // [rsp+10h] [rbp-C8h]
  char *v53; // [rsp+18h] [rbp-C0h]
  __int64 v54; // [rsp+20h] [rbp-B8h]
  __int64 v55; // [rsp+30h] [rbp-A8h]
  int v56; // [rsp+4Ch] [rbp-8Ch] BYREF
  _QWORD *v57; // [rsp+50h] [rbp-88h] BYREF
  unsigned __int64 v58; // [rsp+58h] [rbp-80h] BYREF
  unsigned __int64 v59; // [rsp+60h] [rbp-78h] BYREF
  unsigned __int64 v60; // [rsp+68h] [rbp-70h] BYREF
  __int64 v61; // [rsp+70h] [rbp-68h] BYREF
  __int64 v62; // [rsp+78h] [rbp-60h]
  __int64 v63; // [rsp+80h] [rbp-58h]
  __int64 v64; // [rsp+88h] [rbp-50h]
  unsigned __int64 v65; // [rsp+90h] [rbp-48h]
  char v66; // [rsp+98h] [rbp-40h]
  char v67; // [rsp+99h] [rbp-3Fh]

  v51 = a2;
  v57 = 0;
  v56 = 0;
  if ( a1 != 1 )
    return 3;
  v5 = a5;
  v52 = a3 - 0x474E5543432B2B00LL;
  v50 = (unsigned __int64)(a3 - 0x474E5543432B2B00LL) <= 1;
  if ( v50 && a2 == 6 )
  {
    v6 = *((_QWORD *)a4 - 2);
    if ( v6 )
    {
      v7 = *((_DWORD *)a4 - 9);
      v53 = (char *)*((_QWORD *)a4 - 3);
LABEL_5:
      if ( v7 < 0 )
      {
        sub_2252D40(v5, v53, (__int64)&v61);
        v63 = sub_2252CC0(v66, v5);
        *((_QWORD *)a4 - 2) = sub_2252CC0(v66, v5);
      }
      goto LABEL_7;
    }
    goto LABEL_21;
  }
  v53 = (char *)sub_39F7FC0(a5);
  if ( !v53 )
    return 8;
  v9 = sub_2252D40(v5, v53, (__int64)&v61);
  v63 = sub_2252CC0(v66, v5);
  v10 = sub_39F7F90(v5, &v56);
  v11 = v10 - (v56 == 0);
  if ( v65 <= (unsigned __int64)v9 )
  {
LABEL_16:
    if ( (v51 & 1) != 0 )
    {
      v6 = 0;
      v27 = 0;
      v7 = 0;
LABEL_18:
      result = 6;
      if ( v52 <= 1 )
      {
LABEL_19:
        *((_QWORD *)a4 - 3) = v53;
        v28 = v57;
        *((_DWORD *)a4 - 9) = v7;
        *((_QWORD *)a4 - 1) = v28;
        *((_QWORD *)a4 - 4) = v27;
        *((_QWORD *)a4 - 2) = v6;
        return 6;
      }
      return result;
    }
    if ( (v51 & 8) != 0 || v52 > 1 )
      sub_2207530();
LABEL_21:
    sub_2257630(a4);
  }
  v55 = v5;
  while ( 1 )
  {
    v12 = v67;
    v13 = (char *)sub_2252CC0(v67, 0);
    v14 = sub_2252A40(v12, v13, v9, &v58);
    v15 = v67;
    v16 = v14;
    v17 = (char *)sub_2252CC0(v67, 0);
    v18 = sub_2252A40(v15, v17, v16, &v59);
    v19 = v67;
    v20 = v18;
    v21 = (char *)sub_2252CC0(v67, 0);
    v22 = sub_2252A40(v19, v21, v20, &v60);
    v23 = 0;
    v24 = 0;
    v9 = v22;
    do
    {
      v25 = *v9++;
      v26 = (unsigned __int64)(v25 & 0x7F) << v24;
      v24 += 7;
      v23 |= v26;
    }
    while ( v25 < 0 );
    if ( v61 + v58 > v11 )
      goto LABEL_16;
    if ( v59 + v61 + v58 > v11 )
      break;
    if ( v65 <= (unsigned __int64)v9 )
      goto LABEL_16;
  }
  v5 = v55;
  if ( !v60 )
    return 8;
  v6 = v62 + v60;
  if ( !v23 )
  {
    if ( !v6 )
      return 8;
LABEL_31:
    v7 = v51 & 1;
    if ( (v51 & 1) == 0 )
      goto LABEL_7;
    return 8;
  }
  if ( !v6 )
    return 8;
  if ( v65 + v23 == 1 )
    goto LABEL_31;
  if ( (v51 & 8) != 0 )
  {
    v29 = &`typeinfo for'__cxxabiv1::__forced_unwind;
  }
  else if ( v52 <= 1 )
  {
    v48 = a4 + 32;
    if ( (*a4 & 1) != 0 )
      v48 = (_QWORD *)*((_QWORD *)a4 - 10);
    v57 = v48;
    v29 = (__int64 **)*(v48 - 14);
  }
  else
  {
    v29 = &`typeinfo for'__cxxabiv1::__foreign_exception;
  }
  v30 = 0;
  v31 = v65 + v23 - 1;
  v32 = (__int64)v29;
  v54 = v62 + v60;
  while ( 1 )
  {
    v33 = (char *)v31;
    v34 = 0;
    v35 = 0;
    do
    {
      v36 = *v33++;
      v37 = (unsigned __int64)(v36 & 0x7F) << v35;
      v35 += 7;
      v34 |= v37;
    }
    while ( v36 < 0 );
    if ( v35 <= 0x3F && (v36 & 0x40) != 0 )
      v34 |= -(1LL << v35);
    v38 = v33;
    v39 = 0;
    v40 = 0;
    do
    {
      v41 = *v38++;
      v42 = (unsigned __int64)(v41 & 0x7F) << v40;
      v40 += 7;
      v39 |= v42;
    }
    while ( v41 < 0 );
    if ( v40 <= 0x3F && (v41 & 0x40) != 0 )
      v39 |= -(1LL << v40);
    if ( !v34 )
    {
      v30 = 1;
      goto LABEL_56;
    }
    if ( v34 <= 0 )
      break;
    v46 = v34;
    v43 = sub_2252B90((__int64)&v61, v34);
    if ( !v43 || v32 && (v46 = v32, (unsigned __int8)sub_22529E0(v43, v32, &v57)) )
    {
LABEL_62:
      v27 = v31;
      v5 = v55;
      v7 = v34;
      v6 = v54;
      if ( (v51 & 1) != 0 )
        goto LABEL_18;
      goto LABEL_63;
    }
LABEL_56:
    if ( !v39 )
    {
      v6 = v54;
      v5 = v55;
      if ( v30 )
        goto LABEL_31;
      return 8;
    }
    v31 = (__int64)&v33[v39];
  }
  if ( (v51 & 8) != 0 || !v50 || !v32 )
  {
    v43 = 0;
    LODWORD(v44) = 0;
    v45 = (unsigned __int8 *)(v64 + ~v34);
    do
    {
      v46 = *v45++;
      v47 = (v46 & 0x7F) << v44;
      v44 = (unsigned int)(v44 + 7);
      v43 |= v47;
    }
    while ( (v46 & 0x80u) != 0LL );
    if ( !v43 )
      goto LABEL_62;
    goto LABEL_56;
  }
  v43 = (unsigned __int64)&v61;
  v46 = v32;
  if ( (unsigned __int8)sub_2252C30((__int64)&v61, v32, v57, v34) )
    goto LABEL_56;
  v27 = v31;
  v5 = v55;
  v7 = v34;
  v6 = v54;
  if ( (v51 & 1) != 0 )
    goto LABEL_19;
LABEL_63:
  if ( (v51 & 8) == 0 && v52 <= 1 )
    goto LABEL_5;
  if ( v7 < 0 )
    sub_2207570(v43, v46, v45, v44, v27);
LABEL_7:
  sub_39F7F20(v5, 0, a4);
  sub_39F7F20(v5, 1, v7);
  sub_39F7FB0(v5, v6);
  return 7;
}
