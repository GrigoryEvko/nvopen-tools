// Function: sub_246A570
// Address: 0x246a570
//
void __fastcall sub_246A570(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r14
  __int64 v3; // rax
  __int64 v4; // rax
  _BYTE *v5; // r14
  __int64 v6; // rax
  _BYTE *v7; // rax
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rsi
  unsigned int v13; // r15d
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r15
  unsigned int v19; // ecx
  unsigned int v20; // eax
  __int64 v21; // rax
  _QWORD **v22; // rax
  unsigned __int64 v23; // rax
  __int64 v24; // rdx
  unsigned int v25; // r14d
  __int64 v26; // rax
  __int64 v27; // r14
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 **v30; // r14
  __int64 v31; // rax
  _BYTE *v32; // rax
  unsigned __int64 v33; // rax
  int v34; // edx
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rax
  __int64 v38; // r15
  __int64 v39; // r14
  __int64 v40; // rcx
  __int64 v41; // rdx
  __int64 v42; // r15
  __int64 v43; // r14
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // r14
  unsigned __int64 v47; // r15
  __int64 v48; // rax
  __int64 **v49; // r14
  __int64 v50; // rax
  _BYTE *v51; // rax
  unsigned __int64 v52; // rax
  int v53; // edx
  __int64 v54; // rax
  __int64 v55; // rsi
  __int64 v56; // r14
  __int64 v57; // rcx
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // r15
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 *v63; // rax
  __int64 v64; // rax
  unsigned int v65; // ecx
  _QWORD **v66; // [rsp-1A8h] [rbp-1A8h]
  __int64 v67; // [rsp-1A0h] [rbp-1A0h]
  __int64 v68; // [rsp-198h] [rbp-198h]
  __int64 v69; // [rsp-190h] [rbp-190h]
  _BYTE *v70; // [rsp-190h] [rbp-190h]
  __int64 v71; // [rsp-190h] [rbp-190h]
  _BYTE *v72; // [rsp-190h] [rbp-190h]
  __int64 v73; // [rsp-190h] [rbp-190h]
  __int64 v74; // [rsp-190h] [rbp-190h]
  unsigned __int16 v75; // [rsp-184h] [rbp-184h]
  unsigned __int16 v76; // [rsp-182h] [rbp-182h]
  _QWORD **v77; // [rsp-170h] [rbp-170h]
  int v78; // [rsp-160h] [rbp-160h]
  _BYTE v79[32]; // [rsp-158h] [rbp-158h] BYREF
  __int16 v80; // [rsp-138h] [rbp-138h]
  _DWORD v81[8]; // [rsp-128h] [rbp-128h] BYREF
  __int16 v82; // [rsp-108h] [rbp-108h]
  _BYTE v83[32]; // [rsp-F8h] [rbp-F8h] BYREF
  __int16 v84; // [rsp-D8h] [rbp-D8h]
  unsigned int *v85[2]; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v86; // [rsp-B8h] [rbp-B8h] BYREF
  _QWORD *v87; // [rsp-80h] [rbp-80h]
  void *v88; // [rsp-48h] [rbp-48h]

  if ( *(_DWORD *)(a1 + 40) )
  {
    sub_23D0AB0((__int64)v85, *(_QWORD *)(*(_QWORD *)(a1 + 24) + 480LL), 0, 0, 0);
    v1 = *(_QWORD *)(a1 + 16);
    v84 = 257;
    v2 = *(_QWORD *)(v1 + 152);
    v3 = sub_BCB2E0(v87);
    v4 = sub_A82CA0(v85, v3, v2, 0, 0, (__int64)v83);
    *(_QWORD *)(a1 + 200) = v4;
    v5 = (_BYTE *)v4;
    v6 = *(_QWORD *)(a1 + 16);
    v84 = 257;
    v7 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v6 + 80), 160, 0);
    v8 = sub_929C50(v85, v7, v5, (__int64)v83, 0, 0);
    v9 = *(_QWORD *)(a1 + 16);
    v84 = 257;
    v10 = (__int64 *)sub_BCB2B0(*(_QWORD **)(v9 + 72));
    v11 = sub_23DEB90((__int64 *)v85, v10, v8, (__int64)v83);
    v12 = (unsigned __int8)byte_4FE8EA8;
    *(_QWORD *)(a1 + 184) = v11;
    *(_WORD *)(v11 + 2) = v12 | *(_WORD *)(v11 + 2) & 0xFFC0;
    LODWORD(v11) = (unsigned __int8)v12;
    BYTE1(v11) = 1;
    v13 = v11;
    v14 = sub_BCB2B0(v87);
    v15 = sub_AD6530(v14, v12);
    sub_B34240((__int64)v85, *(_QWORD *)(a1 + 184), v15, v8, v13, 0, 0, 0, 0);
    v16 = *(_QWORD *)(a1 + 16);
    v81[1] = 0;
    v84 = 257;
    v17 = sub_AD64C0(*(_QWORD *)(v16 + 80), 800, 0);
    v18 = sub_B33C40((__int64)v85, 0x16Eu, v8, v17, v81[0], (__int64)v83);
    v19 = (unsigned __int8)byte_4FE8EA8;
    v20 = (unsigned __int8)byte_4FE8EA8;
    BYTE1(v19) = 1;
    BYTE1(v20) = 1;
    sub_B343C0(
      (__int64)v85,
      0xEEu,
      *(_QWORD *)(a1 + 184),
      v19,
      *(_QWORD *)(*(_QWORD *)(a1 + 16) + 136LL),
      v20,
      v18,
      0,
      0,
      0,
      0,
      0);
    v21 = *(_QWORD *)(a1 + 16);
    if ( *(_DWORD *)(v21 + 4) )
    {
      v84 = 257;
      v63 = (__int64 *)sub_BCB2B0(*(_QWORD **)(v21 + 72));
      v64 = sub_23DEB90((__int64 *)v85, v63, v8, (__int64)v83);
      LOWORD(v65) = (unsigned __int8)byte_4FE8EA8;
      *(_QWORD *)(a1 + 192) = v64;
      *(_WORD *)(v64 + 2) = v65 | *(_WORD *)(v64 + 2) & 0xFFC0;
      v65 = (unsigned __int8)v65;
      BYTE1(v65) = 1;
      LODWORD(v64) = 256;
      LOBYTE(v64) = v65;
      sub_B343C0(
        (__int64)v85,
        0xEEu,
        *(_QWORD *)(a1 + 192),
        v65,
        *(_QWORD *)(*(_QWORD *)(a1 + 16) + 144LL),
        v64,
        v18,
        0,
        0,
        0,
        0,
        0);
    }
    sub_F94A20(v85, 238);
    v22 = *(_QWORD ***)(a1 + 32);
    v66 = &v22[*(unsigned int *)(a1 + 40)];
    if ( v66 != v22 )
    {
      v77 = *(_QWORD ***)(a1 + 32);
      do
      {
        v46 = (__int64)*v77;
        sub_2468350((__int64)v85, *v77);
        v47 = *(_QWORD *)(v46 - 32LL * (*(_DWORD *)(v46 + 4) & 0x7FFFFFF));
        v48 = *(_QWORD *)(a1 + 16);
        v84 = 257;
        v49 = *(__int64 ***)(v48 + 96);
        v82 = 257;
        v72 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v48 + 80), 24, 0);
        v50 = *(_QWORD *)(a1 + 16);
        v80 = 257;
        v51 = (_BYTE *)sub_24633A0((__int64 *)v85, 0x2Fu, v47, *(__int64 ***)(v50 + 80), (__int64)v79, 0, v78, 0);
        v52 = sub_929C50(v85, v51, v72, (__int64)v81, 0, 0);
        v53 = sub_24633A0((__int64 *)v85, 0x30u, v52, v49, (__int64)v83, 0, v78, 0);
        v54 = *(_QWORD *)(a1 + 16);
        v84 = 257;
        v55 = *(_QWORD *)(v54 + 96);
        LODWORD(v54) = v76;
        BYTE1(v54) = 0;
        v76 = (unsigned __int8)v76;
        v56 = sub_A82CA0(v85, v55, v53, v54, 0, (__int64)v83);
        v73 = *(_QWORD *)(a1 + 24);
        v57 = sub_BCB2B0(v87);
        if ( **(_BYTE **)(v73 + 8) )
          v23 = (unsigned __int64)sub_2465B30((__int64 *)v73, v56, (__int64)v85, v57, 1);
        else
          v23 = sub_2463FC0(v73, v56, v85, 0x103u);
        v67 = v24;
        v68 = v23;
        v25 = *(_BYTE *)(a1 + 180) == 0 ? 0x68 : 0;
        v69 = *(_QWORD *)(a1 + 184);
        v26 = sub_BCB2E0(v87);
        v27 = v25 + 56LL;
        v28 = sub_ACD640(v26, v27, 0);
        sub_B343C0((__int64)v85, 0xEEu, v68, 0x103u, v69, 0x103u, v28, 0, 0, 0, 0, 0);
        v29 = *(_QWORD *)(a1 + 16);
        if ( *(_DWORD *)(v29 + 4) )
        {
          v74 = *(_QWORD *)(a1 + 192);
          v58 = sub_BCB2E0(v87);
          v59 = sub_ACD640(v58, v27, 0);
          sub_B343C0((__int64)v85, 0xEEu, v67, 0x103u, v74, 0x103u, v59, 0, 0, 0, 0, 0);
          v29 = *(_QWORD *)(a1 + 16);
        }
        v84 = 257;
        v30 = *(__int64 ***)(v29 + 96);
        v82 = 257;
        v70 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v29 + 80), 16, 0);
        v31 = *(_QWORD *)(a1 + 16);
        v80 = 257;
        v32 = (_BYTE *)sub_24633A0((__int64 *)v85, 0x2Fu, v47, *(__int64 ***)(v31 + 80), (__int64)v79, 0, v78, 0);
        v33 = sub_929C50(v85, v32, v70, (__int64)v81, 0, 0);
        v34 = sub_24633A0((__int64 *)v85, 0x30u, v33, v30, (__int64)v83, 0, v78, 0);
        v35 = *(_QWORD *)(a1 + 16);
        v84 = 257;
        v36 = *(_QWORD *)(v35 + 96);
        LODWORD(v35) = v75;
        BYTE1(v35) = 0;
        v75 = (unsigned __int8)v75;
        v37 = sub_A82CA0(v85, v36, v34, v35, 0, (__int64)v83);
        v38 = *(_QWORD *)(a1 + 24);
        v39 = v37;
        v40 = sub_BCB2B0(v87);
        if ( **(_BYTE **)(v38 + 8) )
          v42 = (__int64)sub_2465B30((__int64 *)v38, v39, (__int64)v85, v40, 1);
        else
          v42 = sub_2463FC0(v38, v39, v85, 0x103u);
        v43 = v41;
        v84 = 257;
        v71 = *(_QWORD *)(a1 + 184);
        v44 = sub_BCB2B0(v87);
        v45 = sub_94B060(v85, v44, v71, 0xA0u, (__int64)v83);
        sub_B343C0((__int64)v85, 0xEEu, v42, 0x103u, v45, 0x103u, *(_QWORD *)(a1 + 200), 0, 0, 0, 0, 0);
        if ( *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4LL) )
        {
          v60 = *(_QWORD *)(a1 + 192);
          v84 = 257;
          v61 = sub_BCB2B0(v87);
          v62 = sub_94B060(v85, v61, v60, 0xA0u, (__int64)v83);
          sub_B343C0((__int64)v85, 0xEEu, v43, 0x103u, v62, 0x103u, *(_QWORD *)(a1 + 200), 0, 0, 0, 0, 0);
        }
        nullsub_61();
        v88 = &unk_49DA100;
        nullsub_63();
        if ( (__int64 *)v85[0] != &v86 )
          _libc_free((unsigned __int64)v85[0]);
        ++v77;
      }
      while ( v66 != v77 );
    }
  }
}
