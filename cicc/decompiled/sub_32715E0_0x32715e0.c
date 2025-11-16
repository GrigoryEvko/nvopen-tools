// Function: sub_32715E0
// Address: 0x32715e0
//
__int64 __fastcall sub_32715E0(
        __int64 a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int64 a7,
        __int64 a8,
        int a9)
{
  unsigned int v9; // r12d
  __int64 result; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // r8
  unsigned __int16 *v21; // rcx
  int v22; // eax
  __int64 v23; // rbx
  char **v24; // r15
  bool v25; // al
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rsi
  unsigned __int16 *v33; // r13
  int v34; // eax
  __int64 v35; // rbx
  __int64 v36; // r15
  char **v37; // r15
  __int64 v38; // rdx
  unsigned int v39; // edx
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // rcx
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rdx
  __int64 v47; // [rsp+8h] [rbp-98h]
  __int64 v48; // [rsp+8h] [rbp-98h]
  __int64 v49; // [rsp+8h] [rbp-98h]
  unsigned int v50; // [rsp+8h] [rbp-98h]
  unsigned int v51; // [rsp+8h] [rbp-98h]
  const void *v52; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v53; // [rsp+18h] [rbp-88h]
  __int16 v54; // [rsp+20h] [rbp-80h] BYREF
  __int64 v55; // [rsp+28h] [rbp-78h]
  const void *v56; // [rsp+30h] [rbp-70h] BYREF
  __int64 v57; // [rsp+38h] [rbp-68h]
  unsigned __int64 v58; // [rsp+40h] [rbp-60h] BYREF
  __int64 v59; // [rsp+48h] [rbp-58h]
  __int64 v60; // [rsp+50h] [rbp-50h]
  __int64 v61; // [rsp+58h] [rbp-48h]
  __int64 v62; // [rsp+60h] [rbp-40h] BYREF
  __int64 v63; // [rsp+68h] [rbp-38h]

  v9 = a4;
  if ( a1 != a5 || a2 != a6 )
  {
    if ( *(_DWORD *)(a5 + 24) != 216 )
      return 0;
    v12 = *(_QWORD *)(a5 + 40);
    if ( *(_QWORD *)v12 != a1 || *(_DWORD *)(v12 + 8) != a2 )
      return 0;
  }
  v13 = sub_33CF6D0(a3, a4);
  v15 = sub_33DFBC0(v13, v14, 0, 0);
  v16 = sub_33CF6D0(a7, a8);
  v18 = v17;
  v19 = sub_33DFBC0(v16, v17, 0, 0);
  v20 = v19;
  if ( !v15 || !v19 )
    return 0;
  v21 = (unsigned __int16 *)(*(_QWORD *)(a3 + 48) + 16LL * v9);
  v22 = *v21;
  v23 = *((_QWORD *)v21 + 1);
  v24 = (char **)(*(_QWORD *)(v15 + 96) + 24LL);
  LOWORD(v58) = v22;
  v59 = v23;
  if ( (_WORD)v22 )
  {
    if ( (unsigned __int16)(v22 - 17) > 0xD3u )
    {
      LOWORD(v62) = v22;
      v63 = v23;
      goto LABEL_30;
    }
    LOWORD(v22) = word_4456580[v22 - 1];
    v45 = 0;
  }
  else
  {
    v47 = v20;
    v25 = sub_30070B0((__int64)&v58);
    v20 = v47;
    if ( !v25 )
    {
      v63 = v23;
      LOWORD(v62) = 0;
LABEL_13:
      v48 = v20;
      v28 = sub_3007260((__int64)&v62);
      v20 = v48;
      v29 = v28;
      v30 = v31;
      v60 = v29;
      LODWORD(v31) = v29;
      v61 = v30;
      goto LABEL_14;
    }
    LOWORD(v22) = sub_3009970((__int64)&v58, v18, v26, v27, v47);
    v20 = v47;
  }
  LOWORD(v62) = v22;
  v63 = v45;
  if ( !(_WORD)v22 )
    goto LABEL_13;
LABEL_30:
  if ( (_WORD)v22 == 1 || (unsigned __int16)(v22 - 504) <= 7u )
    goto LABEL_58;
  v31 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v22 - 16];
LABEL_14:
  v32 = (__int64)v24;
  v49 = v20;
  sub_C44740((__int64)&v52, v24, v31);
  v33 = (unsigned __int16 *)(*(_QWORD *)(a7 + 48) + 16LL * (unsigned int)a8);
  v34 = *v33;
  v35 = *((_QWORD *)v33 + 1);
  v36 = *(_QWORD *)(v49 + 96);
  v54 = v34;
  v55 = v35;
  v37 = (char **)(v36 + 24);
  if ( (_WORD)v34 )
  {
    if ( (unsigned __int16)(v34 - 17) > 0xD3u )
    {
      LOWORD(v56) = v34;
      v57 = v35;
LABEL_17:
      if ( (_WORD)v34 != 1 && (unsigned __int16)(v34 - 504) > 7u )
      {
        v38 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v34 - 16];
        goto LABEL_20;
      }
LABEL_58:
      BUG();
    }
    LOWORD(v34) = word_4456580[v34 - 1];
    v46 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v54) )
    {
      v57 = v35;
      LOWORD(v56) = 0;
      goto LABEL_41;
    }
    LOWORD(v34) = sub_3009970((__int64)&v54, v32, v40, v41, v42);
  }
  LOWORD(v56) = v34;
  v57 = v46;
  if ( (_WORD)v34 )
    goto LABEL_17;
LABEL_41:
  v43 = sub_3007260((__int64)&v56);
  v44 = v38;
  v62 = v43;
  LODWORD(v38) = v43;
  v63 = v44;
LABEL_20:
  sub_C44740((__int64)&v58, v37, v38);
  v39 = v59;
  if ( v53 >= (unsigned int)v59 )
  {
    sub_C44830((__int64)&v56, &v58, v53);
    if ( v53 <= 0x40 )
    {
      if ( v52 != v56 )
      {
LABEL_35:
        if ( (unsigned int)v57 > 0x40 && v56 )
          j_j___libc_free_0_0((unsigned __int64)v56);
        v39 = v59;
        goto LABEL_21;
      }
    }
    else if ( !sub_C43C50((__int64)&v52, &v56) )
    {
      goto LABEL_35;
    }
    if ( (unsigned int)v57 > 0x40 && v56 )
      j_j___libc_free_0_0((unsigned __int64)v56);
    v39 = v59;
    result = 180;
    if ( a9 != 20 )
    {
      result = 181;
      if ( a9 != 18 )
        result = 0;
    }
    goto LABEL_22;
  }
LABEL_21:
  result = 0;
LABEL_22:
  if ( v39 > 0x40 && v58 )
  {
    v50 = result;
    j_j___libc_free_0_0(v58);
    result = v50;
  }
  if ( v53 > 0x40 )
  {
    if ( v52 )
    {
      v51 = result;
      j_j___libc_free_0_0((unsigned __int64)v52);
      return v51;
    }
  }
  return result;
}
