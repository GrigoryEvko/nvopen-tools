// Function: sub_375FFB0
// Address: 0x375ffb0
//
__int64 __fastcall sub_375FFB0(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned __int64 a7,
        __int64 a8)
{
  _BYTE *v9; // rax
  __int64 v10; // r15
  __int64 v11; // rax
  unsigned __int16 v12; // dx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned int v16; // eax
  __int64 v17; // r15
  __int64 v18; // rax
  unsigned __int16 v19; // dx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  unsigned __int16 v24; // dx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdx
  unsigned int v28; // eax
  __int64 v29; // rax
  unsigned __int16 v30; // dx
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rdx
  unsigned int v34; // eax
  __int64 v35; // rbx
  __int64 v36; // rax
  unsigned __int16 v37; // dx
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rax
  unsigned __int16 v42; // dx
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rdx
  unsigned int v46; // eax
  _DWORD *v47; // rbx
  __int64 result; // rax
  unsigned int v49; // [rsp+8h] [rbp-108h]
  unsigned int v50; // [rsp+8h] [rbp-108h]
  unsigned __int64 v51; // [rsp+10h] [rbp-100h] BYREF
  __int64 v52; // [rsp+18h] [rbp-F8h]
  __int64 v53; // [rsp+20h] [rbp-F0h] BYREF
  char v54; // [rsp+28h] [rbp-E8h]
  __int64 v55; // [rsp+30h] [rbp-E0h] BYREF
  char v56; // [rsp+38h] [rbp-D8h]
  unsigned __int16 v57; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v58; // [rsp+48h] [rbp-C8h]
  unsigned __int16 v59; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v60; // [rsp+58h] [rbp-B8h]
  unsigned __int16 v61; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v62; // [rsp+68h] [rbp-A8h]
  __int64 v63; // [rsp+70h] [rbp-A0h]
  __int64 v64; // [rsp+78h] [rbp-98h]
  __int64 v65; // [rsp+80h] [rbp-90h]
  __int64 v66; // [rsp+88h] [rbp-88h]
  int v67; // [rsp+90h] [rbp-80h] BYREF
  __int64 v68; // [rsp+98h] [rbp-78h]
  __int64 v69; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v70; // [rsp+A8h] [rbp-68h]
  __int64 v71; // [rsp+B0h] [rbp-60h]
  __int64 v72; // [rsp+B8h] [rbp-58h]
  __int64 v73; // [rsp+C0h] [rbp-50h]
  __int64 v74; // [rsp+C8h] [rbp-48h]
  __int64 v75; // [rsp+D0h] [rbp-40h] BYREF
  __int64 v76; // [rsp+D8h] [rbp-38h]

  v51 = a4;
  v52 = a5;
  sub_375EAB0(a1, (__int64)&v51);
  sub_375EAB0(a1, (__int64)&a7);
  v9 = (_BYTE *)sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 8) + 40LL));
  v10 = *(_QWORD *)(a1 + 8);
  if ( !*v9 )
  {
    v29 = *(_QWORD *)(v51 + 48) + 16LL * (unsigned int)v52;
    v30 = *(_WORD *)v29;
    v31 = *(_QWORD *)(v29 + 8);
    LOWORD(v75) = v30;
    v76 = v31;
    if ( v30 )
    {
      if ( v30 == 1 || (unsigned __int16)(v30 - 504) <= 7u )
        goto LABEL_35;
      v33 = 16LL * (v30 - 1);
      v32 = *(_QWORD *)&byte_444C4A0[v33];
      LOBYTE(v33) = byte_444C4A0[v33 + 8];
    }
    else
    {
      v32 = sub_3007260((__int64)&v75);
      v71 = v32;
      v72 = v33;
    }
    LOBYTE(v76) = v33;
    v75 = v32;
    v34 = sub_CA1930(&v75);
    sub_33F9B80(v10, a2, a3, v51, v52, 0, v34, 0);
    v35 = *(_QWORD *)(a1 + 8);
    v36 = *(_QWORD *)(a7 + 48) + 16LL * (unsigned int)a8;
    v37 = *(_WORD *)v36;
    v38 = *(_QWORD *)(v36 + 8);
    LOWORD(v69) = v37;
    v70 = v38;
    if ( v37 )
    {
      if ( v37 == 1 || (unsigned __int16)(v37 - 504) <= 7u )
        goto LABEL_35;
      v40 = 16LL * (v37 - 1);
      v39 = *(_QWORD *)&byte_444C4A0[v40];
      LOBYTE(v40) = byte_444C4A0[v40 + 8];
    }
    else
    {
      v39 = sub_3007260((__int64)&v69);
      v73 = v39;
      v74 = v40;
    }
    v56 = v40;
    v55 = v39;
    v50 = sub_CA1930(&v55);
    v41 = *(_QWORD *)(v51 + 48) + 16LL * (unsigned int)v52;
    v42 = *(_WORD *)v41;
    v43 = *(_QWORD *)(v41 + 8);
    LOWORD(v67) = v42;
    v68 = v43;
    if ( v42 )
    {
      if ( v42 == 1 || (unsigned __int16)(v42 - 504) <= 7u )
        goto LABEL_35;
      v45 = 16LL * (v42 - 1);
      v44 = *(_QWORD *)&byte_444C4A0[v45];
      LOBYTE(v45) = byte_444C4A0[v45 + 8];
    }
    else
    {
      v44 = sub_3007260((__int64)&v67);
      v75 = v44;
      v76 = v45;
    }
    LOBYTE(v70) = v45;
    v69 = v44;
    v46 = sub_CA1930(&v69);
    sub_33F9B80(v35, a2, a3, a7, a8, v46, v50, 1);
    goto LABEL_16;
  }
  v11 = *(_QWORD *)(a7 + 48) + 16LL * (unsigned int)a8;
  v12 = *(_WORD *)v11;
  v13 = *(_QWORD *)(v11 + 8);
  v61 = v12;
  v62 = v13;
  if ( v12 )
  {
    if ( v12 == 1 || (unsigned __int16)(v12 - 504) <= 7u )
      goto LABEL_35;
    v15 = 16LL * (v12 - 1);
    v14 = *(_QWORD *)&byte_444C4A0[v15];
    LOBYTE(v15) = byte_444C4A0[v15 + 8];
  }
  else
  {
    v14 = sub_3007260((__int64)&v61);
    v63 = v14;
    v64 = v15;
  }
  LOBYTE(v76) = v15;
  v75 = v14;
  v16 = sub_CA1930(&v75);
  sub_33F9B80(v10, a2, a3, a7, a8, 0, v16, 0);
  v17 = *(_QWORD *)(a1 + 8);
  v18 = *(_QWORD *)(v51 + 48) + 16LL * (unsigned int)v52;
  v19 = *(_WORD *)v18;
  v20 = *(_QWORD *)(v18 + 8);
  v59 = v19;
  v60 = v20;
  if ( v19 )
  {
    if ( v19 == 1 || (unsigned __int16)(v19 - 504) <= 7u )
      goto LABEL_35;
    v22 = 16LL * (v19 - 1);
    v21 = *(_QWORD *)&byte_444C4A0[v22];
    LOBYTE(v22) = byte_444C4A0[v22 + 8];
  }
  else
  {
    v21 = sub_3007260((__int64)&v59);
    v65 = v21;
    v66 = v22;
  }
  v54 = v22;
  v53 = v21;
  v49 = sub_CA1930(&v53);
  v23 = *(_QWORD *)(a7 + 48) + 16LL * (unsigned int)a8;
  v24 = *(_WORD *)v23;
  v25 = *(_QWORD *)(v23 + 8);
  v57 = v24;
  v58 = v25;
  if ( v24 )
  {
    if ( v24 != 1 && (unsigned __int16)(v24 - 504) > 7u )
    {
      v27 = 16LL * (v24 - 1);
      v26 = *(_QWORD *)&byte_444C4A0[v27];
      LOBYTE(v27) = byte_444C4A0[v27 + 8];
      goto LABEL_8;
    }
LABEL_35:
    BUG();
  }
  v26 = sub_3007260((__int64)&v57);
  v69 = v26;
  v70 = v27;
LABEL_8:
  LOBYTE(v76) = v27;
  v75 = v26;
  v28 = sub_CA1930(&v75);
  sub_33F9B80(v17, a2, a3, v51, v52, v28, v49, 1);
LABEL_16:
  v67 = sub_375D5B0(a1, a2, a3);
  v47 = sub_375C600(a1 + 792, &v67);
  *v47 = sub_375D5B0(a1, v51, v52);
  result = sub_375D5B0(a1, a7, a8);
  v47[1] = result;
  return result;
}
