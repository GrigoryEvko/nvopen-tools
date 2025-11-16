// Function: sub_2468CF0
// Address: 0x2468cf0
//
void __fastcall sub_2468CF0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r14
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rsi
  _BYTE *v6; // r14
  __int64 v7; // rax
  _BYTE *v8; // rax
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 *v11; // rax
  __int64 v12; // rax
  __int64 v13; // rsi
  unsigned int v14; // r15d
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // r15
  unsigned int v20; // ecx
  unsigned int v21; // eax
  __int64 v22; // rax
  _QWORD **v23; // rax
  unsigned __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rsi
  __int64 v27; // rax
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
  unsigned int v44; // ecx
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // r14
  unsigned __int64 v48; // r15
  __int64 v49; // rax
  __int64 **v50; // r14
  __int64 v51; // rax
  _BYTE *v52; // rax
  unsigned __int64 v53; // rax
  int v54; // edx
  __int64 v55; // rax
  __int64 v56; // rsi
  __int64 v57; // r14
  __int64 v58; // rcx
  __int64 v59; // r14
  __int64 v60; // rsi
  __int64 v61; // rax
  __int64 v62; // rax
  unsigned int v63; // ecx
  __int64 v64; // r15
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 *v67; // rax
  __int64 v68; // rax
  unsigned int v69; // ecx
  _QWORD **v70; // [rsp-1A8h] [rbp-1A8h]
  __int64 v71; // [rsp-1A0h] [rbp-1A0h]
  unsigned int v72; // [rsp-198h] [rbp-198h]
  __int64 v73; // [rsp-190h] [rbp-190h]
  _BYTE *v74; // [rsp-190h] [rbp-190h]
  __int64 v75; // [rsp-190h] [rbp-190h]
  _BYTE *v76; // [rsp-190h] [rbp-190h]
  __int64 v77; // [rsp-190h] [rbp-190h]
  __int64 v78; // [rsp-190h] [rbp-190h]
  unsigned int v79; // [rsp-190h] [rbp-190h]
  unsigned __int16 v80; // [rsp-184h] [rbp-184h]
  unsigned __int16 v81; // [rsp-182h] [rbp-182h]
  _QWORD **v82; // [rsp-170h] [rbp-170h]
  int v83; // [rsp-160h] [rbp-160h]
  _BYTE v84[32]; // [rsp-158h] [rbp-158h] BYREF
  __int16 v85; // [rsp-138h] [rbp-138h]
  _DWORD v86[8]; // [rsp-128h] [rbp-128h] BYREF
  __int16 v87; // [rsp-108h] [rbp-108h]
  _BYTE v88[32]; // [rsp-F8h] [rbp-F8h] BYREF
  __int16 v89; // [rsp-D8h] [rbp-D8h]
  unsigned int *v90[2]; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v91; // [rsp-B8h] [rbp-B8h] BYREF
  _QWORD *v92; // [rsp-80h] [rbp-80h]
  void *v93; // [rsp-48h] [rbp-48h]

  if ( *(_DWORD *)(a1 + 40) )
  {
    sub_23D0AB0((__int64)v90, *(_QWORD *)(*(_QWORD *)(a1 + 24) + 480LL), 0, 0, 0);
    v1 = *(_QWORD *)(a1 + 16);
    v89 = 257;
    v2 = *(_QWORD *)(v1 + 152);
    v3 = sub_BCB2E0(v92);
    v4 = sub_A82CA0(v90, v3, v2, 0, 0, (__int64)v88);
    v5 = *(unsigned int *)(a1 + 180);
    *(_QWORD *)(a1 + 200) = v4;
    v6 = (_BYTE *)v4;
    v7 = *(_QWORD *)(a1 + 16);
    v89 = 257;
    v8 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v7 + 80), v5, 0);
    v9 = sub_929C50(v90, v8, v6, (__int64)v88, 0, 0);
    v10 = *(_QWORD *)(a1 + 16);
    v89 = 257;
    v11 = (__int64 *)sub_BCB2B0(*(_QWORD **)(v10 + 72));
    v12 = sub_23DEB90((__int64 *)v90, v11, v9, (__int64)v88);
    v13 = (unsigned __int8)byte_4FE8EA8;
    *(_QWORD *)(a1 + 184) = v12;
    *(_WORD *)(v12 + 2) = v13 | *(_WORD *)(v12 + 2) & 0xFFC0;
    LODWORD(v12) = (unsigned __int8)v13;
    BYTE1(v12) = 1;
    v14 = v12;
    v15 = sub_BCB2B0(v92);
    v16 = sub_AD6530(v15, v13);
    sub_B34240((__int64)v90, *(_QWORD *)(a1 + 184), v16, v9, v14, 0, 0, 0, 0);
    v17 = *(_QWORD *)(a1 + 16);
    v86[1] = 0;
    v89 = 257;
    v18 = sub_AD64C0(*(_QWORD *)(v17 + 80), 800, 0);
    v19 = sub_B33C40((__int64)v90, 0x16Eu, v9, v18, v86[0], (__int64)v88);
    v20 = (unsigned __int8)byte_4FE8EA8;
    v21 = (unsigned __int8)byte_4FE8EA8;
    BYTE1(v20) = 1;
    BYTE1(v21) = 1;
    sub_B343C0(
      (__int64)v90,
      0xEEu,
      *(_QWORD *)(a1 + 184),
      v20,
      *(_QWORD *)(*(_QWORD *)(a1 + 16) + 136LL),
      v21,
      v19,
      0,
      0,
      0,
      0,
      0);
    v22 = *(_QWORD *)(a1 + 16);
    if ( *(_DWORD *)(v22 + 4) )
    {
      v89 = 257;
      v67 = (__int64 *)sub_BCB2B0(*(_QWORD **)(v22 + 72));
      v68 = sub_23DEB90((__int64 *)v90, v67, v9, (__int64)v88);
      LOWORD(v69) = (unsigned __int8)byte_4FE8EA8;
      *(_QWORD *)(a1 + 192) = v68;
      *(_WORD *)(v68 + 2) = v69 | *(_WORD *)(v68 + 2) & 0xFFC0;
      v69 = (unsigned __int8)v69;
      BYTE1(v69) = 1;
      LODWORD(v68) = 256;
      LOBYTE(v68) = v69;
      sub_B343C0(
        (__int64)v90,
        0xEEu,
        *(_QWORD *)(a1 + 192),
        v69,
        *(_QWORD *)(*(_QWORD *)(a1 + 16) + 144LL),
        v68,
        v19,
        0,
        0,
        0,
        0,
        0);
    }
    sub_F94A20(v90, 238);
    v23 = *(_QWORD ***)(a1 + 32);
    v70 = &v23[*(unsigned int *)(a1 + 40)];
    if ( v70 != v23 )
    {
      v82 = *(_QWORD ***)(a1 + 32);
      do
      {
        v47 = (__int64)*v82;
        sub_2468350((__int64)v90, *v82);
        v48 = *(_QWORD *)(v47 - 32LL * (*(_DWORD *)(v47 + 4) & 0x7FFFFFF));
        v49 = *(_QWORD *)(a1 + 16);
        v89 = 257;
        v50 = *(__int64 ***)(v49 + 96);
        v87 = 257;
        v76 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v49 + 80), 16, 0);
        v51 = *(_QWORD *)(a1 + 16);
        v85 = 257;
        v52 = (_BYTE *)sub_24633A0((__int64 *)v90, 0x2Fu, v48, *(__int64 ***)(v51 + 80), (__int64)v84, 0, v83, 0);
        v53 = sub_929C50(v90, v52, v76, (__int64)v86, 0, 0);
        v54 = sub_24633A0((__int64 *)v90, 0x30u, v53, v50, (__int64)v88, 0, v83, 0);
        v55 = *(_QWORD *)(a1 + 16);
        v89 = 257;
        v56 = *(_QWORD *)(v55 + 96);
        LODWORD(v55) = v80;
        BYTE1(v55) = 0;
        v80 = (unsigned __int8)v80;
        v57 = sub_A82CA0(v90, v56, v54, v55, 0, (__int64)v88);
        v77 = *(_QWORD *)(a1 + 24);
        v58 = sub_BCB2B0(v92);
        if ( **(_BYTE **)(v77 + 8) )
          v24 = (unsigned __int64)sub_2465B30((__int64 *)v77, v57, (__int64)v90, v58, 1);
        else
          v24 = sub_2463FC0(v77, v57, v90, 0x104u);
        v71 = v25;
        v59 = v24;
        v26 = *(unsigned int *)(a1 + 180);
        v73 = *(_QWORD *)(a1 + 184);
        v27 = sub_BCB2E0(v92);
        v28 = sub_ACD640(v27, v26, 0);
        sub_B343C0((__int64)v90, 0xEEu, v59, 0x104u, v73, 0x104u, v28, 0, 0, 0, 0, 0);
        v29 = *(_QWORD *)(a1 + 16);
        if ( *(_DWORD *)(v29 + 4) )
        {
          v60 = *(unsigned int *)(a1 + 180);
          v78 = *(_QWORD *)(a1 + 192);
          v61 = sub_BCB2E0(v92);
          v62 = sub_ACD640(v61, v60, 0);
          sub_B343C0((__int64)v90, 0xEEu, v71, 0x104u, v78, 0x104u, v62, 0, 0, 0, 0, 0);
          v29 = *(_QWORD *)(a1 + 16);
        }
        v89 = 257;
        v30 = *(__int64 ***)(v29 + 96);
        v87 = 257;
        v74 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v29 + 80), 8, 0);
        v31 = *(_QWORD *)(a1 + 16);
        v85 = 257;
        v32 = (_BYTE *)sub_24633A0((__int64 *)v90, 0x2Fu, v48, *(__int64 ***)(v31 + 80), (__int64)v84, 0, v83, 0);
        v33 = sub_929C50(v90, v32, v74, (__int64)v86, 0, 0);
        v34 = sub_24633A0((__int64 *)v90, 0x30u, v33, v30, (__int64)v88, 0, v83, 0);
        v35 = *(_QWORD *)(a1 + 16);
        v89 = 257;
        v36 = *(_QWORD *)(v35 + 96);
        LODWORD(v35) = v81;
        BYTE1(v35) = 0;
        v81 = (unsigned __int8)v81;
        v37 = sub_A82CA0(v90, v36, v34, v35, 0, (__int64)v88);
        v38 = *(_QWORD *)(a1 + 24);
        v39 = v37;
        v40 = sub_BCB2B0(v92);
        if ( **(_BYTE **)(v38 + 8) )
          v42 = (__int64)sub_2465B30((__int64 *)v38, v39, (__int64)v90, v40, 1);
        else
          v42 = sub_2463FC0(v38, v39, v90, 0x104u);
        v43 = v41;
        v44 = *(_DWORD *)(a1 + 180);
        v89 = 257;
        v72 = v44;
        v75 = *(_QWORD *)(a1 + 184);
        v45 = sub_BCB2B0(v92);
        v46 = sub_94B060(v90, v45, v75, v72, (__int64)v88);
        sub_B343C0((__int64)v90, 0xEEu, v42, 0x104u, v46, 0x104u, *(_QWORD *)(a1 + 200), 0, 0, 0, 0, 0);
        if ( *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4LL) )
        {
          v63 = *(_DWORD *)(a1 + 180);
          v64 = *(_QWORD *)(a1 + 192);
          v89 = 257;
          v79 = v63;
          v65 = sub_BCB2B0(v92);
          v66 = sub_94B060(v90, v65, v64, v79, (__int64)v88);
          sub_B343C0((__int64)v90, 0xEEu, v43, 0x104u, v66, 0x104u, *(_QWORD *)(a1 + 200), 0, 0, 0, 0, 0);
        }
        nullsub_61();
        v93 = &unk_49DA100;
        nullsub_63();
        if ( (__int64 *)v90[0] != &v91 )
          _libc_free((unsigned __int64)v90[0]);
        ++v82;
      }
      while ( v70 != v82 );
    }
  }
}
