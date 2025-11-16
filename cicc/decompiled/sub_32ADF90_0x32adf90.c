// Function: sub_32ADF90
// Address: 0x32adf90
//
__int64 __fastcall sub_32ADF90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int16 v4; // bx
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rbx
  unsigned __int16 *v11; // rdx
  int v12; // eax
  _QWORD *v13; // r14
  __int64 v14; // r8
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  unsigned __int64 v18; // rax
  unsigned int v19; // eax
  unsigned __int64 v20; // rdx
  __int64 v21; // r13
  __int64 *v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  unsigned int v27; // eax
  unsigned __int64 v28; // rdx
  unsigned int v29; // eax
  unsigned __int64 v30; // rdx
  bool v31; // bl
  bool v32; // [rsp+8h] [rbp-138h]
  bool v33; // [rsp+8h] [rbp-138h]
  int v34; // [rsp+8h] [rbp-138h]
  __int64 v35; // [rsp+10h] [rbp-130h] BYREF
  __int64 v36; // [rsp+18h] [rbp-128h]
  __int64 v37; // [rsp+20h] [rbp-120h] BYREF
  __int64 v38; // [rsp+28h] [rbp-118h]
  unsigned __int64 v39; // [rsp+30h] [rbp-110h] BYREF
  unsigned int v40; // [rsp+38h] [rbp-108h]
  unsigned __int64 v41; // [rsp+40h] [rbp-100h] BYREF
  unsigned int v42; // [rsp+48h] [rbp-F8h]
  unsigned __int64 v43; // [rsp+50h] [rbp-F0h] BYREF
  unsigned int v44; // [rsp+58h] [rbp-E8h]
  unsigned __int64 v45; // [rsp+60h] [rbp-E0h] BYREF
  _QWORD *v46; // [rsp+68h] [rbp-D8h]
  __int64 v47; // [rsp+70h] [rbp-D0h]
  __int64 v48; // [rsp+78h] [rbp-C8h]
  unsigned __int64 v49; // [rsp+90h] [rbp-B0h] BYREF
  __int64 *v50; // [rsp+98h] [rbp-A8h]
  unsigned __int64 v51; // [rsp+A0h] [rbp-A0h] BYREF
  unsigned int v52; // [rsp+A8h] [rbp-98h]
  __int64 v53; // [rsp+B0h] [rbp-90h]
  unsigned __int64 v54; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v55; // [rsp+C8h] [rbp-78h]
  __int64 *v56; // [rsp+D0h] [rbp-70h]
  unsigned __int64 v57; // [rsp+D8h] [rbp-68h] BYREF
  __int64 v58; // [rsp+E0h] [rbp-60h]
  __int64 v59; // [rsp+E8h] [rbp-58h]
  unsigned __int64 v60; // [rsp+F0h] [rbp-50h] BYREF
  unsigned int v61; // [rsp+F8h] [rbp-48h]
  char v62; // [rsp+104h] [rbp-3Ch]

  v4 = a3;
  v35 = a3;
  v36 = a4;
  if ( (_WORD)a3 )
  {
    if ( (unsigned __int16)(a3 - 17) <= 0xD3u )
    {
      v55 = 0;
      v4 = word_4456580[(unsigned __int16)a3 - 1];
      LOWORD(v54) = v4;
      if ( !v4 )
        goto LABEL_5;
      goto LABEL_53;
    }
    goto LABEL_3;
  }
  if ( !sub_30070B0((__int64)&v35) )
  {
LABEL_3:
    v5 = v36;
    goto LABEL_4;
  }
  v4 = sub_3009970((__int64)&v35, a2, v24, v25, v26);
LABEL_4:
  LOWORD(v54) = v4;
  v55 = v5;
  if ( !v4 )
  {
LABEL_5:
    v6 = sub_3007260((__int64)&v54);
    v10 = v7;
    v8 = v6;
    v9 = v10;
    v47 = v8;
    LODWORD(v10) = v8;
    v48 = v9;
    goto LABEL_6;
  }
LABEL_53:
  if ( v4 == 1 || (unsigned __int16)(v4 - 504) <= 7u )
    goto LABEL_106;
  v10 = *(_QWORD *)&byte_444C4A0[16 * v4 - 16];
LABEL_6:
  v11 = (unsigned __int16 *)(*(_QWORD *)(a1 + 48) + 16LL * (unsigned int)a2);
  v12 = *v11;
  v13 = (_QWORD *)*((_QWORD *)v11 + 1);
  LOWORD(v45) = v12;
  v46 = v13;
  if ( (_WORD)v12 )
  {
    if ( (unsigned __int16)(v12 - 17) > 0xD3u )
    {
      LOWORD(v49) = v12;
      v50 = v13;
LABEL_9:
      if ( (_WORD)v12 != 1 && (unsigned __int16)(v12 - 504) > 7u )
      {
        v14 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v12 - 16];
        goto LABEL_15;
      }
LABEL_106:
      BUG();
    }
    LOWORD(v12) = word_4456580[v12 - 1];
    v23 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v45) )
    {
      v50 = v13;
      LOWORD(v49) = 0;
      goto LABEL_14;
    }
    LOWORD(v12) = sub_3009970((__int64)&v45, a2, v15, v16, v17);
  }
  LOWORD(v49) = v12;
  v50 = v23;
  if ( (_WORD)v12 )
    goto LABEL_9;
LABEL_14:
  LODWORD(v14) = sub_3007260((__int64)&v49);
LABEL_15:
  v37 = 0;
  LODWORD(v38) = 0;
  LODWORD(v55) = v10;
  if ( (unsigned int)v10 > 0x40 )
  {
    v34 = v14;
    sub_C43690((__int64)&v54, -1, 1);
    LODWORD(v14) = v34;
  }
  else
  {
    v18 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v10;
    if ( !(_DWORD)v10 )
      v18 = 0;
    v54 = v18;
  }
  sub_C449B0((__int64)&v39, (const void **)&v54, v14);
  if ( (unsigned int)v55 > 0x40 && v54 )
    j_j___libc_free_0_0(v54);
  LODWORD(v46) = 64;
  v19 = v40;
  v45 = 0;
  v42 = v40;
  if ( v40 <= 0x40 )
  {
    v20 = v39;
    v42 = 0;
    v44 = v40;
    v41 = v39;
    v43 = v39;
    LODWORD(v49) = 180;
    v50 = &v37;
    v52 = v40;
LABEL_24:
    v51 = v20;
    BYTE4(v53) = 0;
    LODWORD(v54) = 181;
    LODWORD(v55) = 180;
    v56 = v50;
    LODWORD(v58) = v19;
LABEL_25:
    v57 = v51;
    goto LABEL_26;
  }
  sub_C43780((__int64)&v41, (const void **)&v39);
  v19 = v42;
  v20 = v41;
  v42 = 0;
  LODWORD(v49) = 180;
  v44 = v19;
  v43 = v41;
  v50 = &v37;
  v52 = v19;
  if ( v19 <= 0x40 )
    goto LABEL_24;
  sub_C43780((__int64)&v51, (const void **)&v43);
  BYTE4(v53) = 0;
  LODWORD(v54) = 181;
  LODWORD(v55) = v49;
  LODWORD(v58) = v52;
  v56 = v50;
  if ( v52 <= 0x40 )
    goto LABEL_25;
  sub_C43780((__int64)&v57, (const void **)&v51);
LABEL_26:
  v59 = v53;
  v61 = (unsigned int)v46;
  if ( (unsigned int)v46 > 0x40 )
    sub_C43780((__int64)&v60, (const void **)&v45);
  else
    v60 = v45;
  v62 = 0;
  v32 = sub_32ADDA0(a1, a2, 0, (__int64)&v54);
  if ( v61 > 0x40 && v60 )
    j_j___libc_free_0_0(v60);
  if ( (unsigned int)v58 > 0x40 && v57 )
    j_j___libc_free_0_0(v57);
  if ( v52 > 0x40 && v51 )
    j_j___libc_free_0_0(v51);
  if ( v44 > 0x40 && v43 )
    j_j___libc_free_0_0(v43);
  if ( v42 > 0x40 && v41 )
    j_j___libc_free_0_0(v41);
  if ( (unsigned int)v46 > 0x40 && v45 )
    j_j___libc_free_0_0(v45);
  if ( v32 )
    goto LABEL_47;
  v27 = v40;
  LODWORD(v46) = v40;
  if ( v40 > 0x40 )
  {
    sub_C43780((__int64)&v45, (const void **)&v39);
    v27 = (unsigned int)v46;
    v28 = v45;
  }
  else
  {
    v28 = v39;
    v45 = v39;
  }
  LODWORD(v46) = 0;
  LODWORD(v50) = v27;
  v49 = v28;
  LODWORD(v54) = 180;
  LODWORD(v55) = 181;
  v56 = &v37;
  v57 = 0;
  v58 = 64;
  BYTE4(v59) = 0;
  v61 = v27;
  if ( v27 > 0x40 )
    sub_C43780((__int64)&v60, (const void **)&v49);
  else
    v60 = v28;
  v62 = 0;
  v33 = sub_32ADDA0(a1, a2, 0, (__int64)&v54);
  if ( v61 > 0x40 && v60 )
    j_j___libc_free_0_0(v60);
  if ( (unsigned int)v58 > 0x40 && v57 )
    j_j___libc_free_0_0(v57);
  if ( (unsigned int)v50 > 0x40 && v49 )
    j_j___libc_free_0_0(v49);
  if ( (unsigned int)v46 > 0x40 && v45 )
    j_j___libc_free_0_0(v45);
  if ( v33 )
    goto LABEL_47;
  v29 = v40;
  LODWORD(v46) = v40;
  if ( v40 > 0x40 )
  {
    sub_C43780((__int64)&v45, (const void **)&v39);
    v29 = (unsigned int)v46;
    v30 = v45;
  }
  else
  {
    v30 = v39;
    v45 = v39;
  }
  LODWORD(v46) = 0;
  LODWORD(v50) = v29;
  v49 = v30;
  LODWORD(v54) = 182;
  LODWORD(v55) = 181;
  v56 = &v37;
  v57 = 0;
  v58 = 64;
  BYTE4(v59) = 0;
  v61 = v29;
  if ( v29 > 0x40 )
    sub_C43780((__int64)&v60, (const void **)&v49);
  else
    v60 = v30;
  v62 = 0;
  v31 = sub_32ADDA0(a1, a2, 0, (__int64)&v54);
  if ( v61 > 0x40 && v60 )
    j_j___libc_free_0_0(v60);
  if ( (unsigned int)v58 > 0x40 && v57 )
    j_j___libc_free_0_0(v57);
  if ( (unsigned int)v50 > 0x40 && v49 )
    j_j___libc_free_0_0(v49);
  if ( (unsigned int)v46 > 0x40 && v45 )
    j_j___libc_free_0_0(v45);
  if ( v31 )
LABEL_47:
    v21 = v37;
  else
    v21 = 0;
  if ( v40 > 0x40 && v39 )
    j_j___libc_free_0_0(v39);
  return v21;
}
