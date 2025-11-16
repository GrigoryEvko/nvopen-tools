// Function: sub_854CA0
// Address: 0x854ca0
//
__int64 sub_854CA0()
{
  __int64 v0; // rax
  __int64 v1; // rdx
  __int64 v2; // rcx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  bool v29; // zf
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rdx
  int v38; // edi
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // r15
  __int64 v52; // rdx
  __int64 v53; // rax
  __int64 v54; // r13
  __int64 v55; // rdx
  __int64 result; // rax
  __int64 v57; // rdx
  __int64 v58; // rax
  __int64 v59; // r13
  __int64 v60; // rdx
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // r13
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rax
  __int64 v73; // r13
  __int64 v74; // rdx

  unk_4D03E90 = 0;
  memset(qword_4D03D40, 0, 0x140u);
  v0 = sub_823970(24);
  v1 = unk_4D03E90;
  v2 = *(_QWORD *)(v0 + 12);
  *(_BYTE *)(v0 + 8) = 1;
  *(_QWORD *)v0 = v1;
  unk_4D03E90 = v0;
  *(_QWORD *)(v0 + 12) = v2 & 0xE0000000000000LL | 0x508011000000001LL;
  qword_4D03D40[1] = v0;
  v3 = sub_823970(24);
  v4 = *(_QWORD *)(v3 + 12);
  *(_BYTE *)(v3 + 8) = 2;
  *(_QWORD *)(v3 + 12) = v4 & 0xE0000000000000LL | 0x508011000000001LL;
  *(_QWORD *)v3 = unk_4D03E90;
  qword_4D03D40[2] = v3;
  unk_4D03E90 = v3;
  v5 = sub_823970(24);
  v6 = *(_QWORD *)(v5 + 12);
  *(_BYTE *)(v5 + 8) = 3;
  *(_QWORD *)v5 = 0;
  *(_QWORD *)(v5 + 12) = v6 & 0xE0000000000000LL | 0x30A010000000001LL;
  qword_4D03D40[3] = v5;
  v7 = sub_823970(24);
  v8 = *(_QWORD *)(v7 + 12);
  *(_BYTE *)(v7 + 8) = 4;
  *(_QWORD *)v7 = 0;
  *(_QWORD *)(v7 + 12) = v8 & 0xE0000000000000LL | 0x30A010000000001LL;
  qword_4D03D40[4] = v7;
  v9 = sub_823970(24);
  v10 = *(_QWORD *)(v9 + 12);
  *(_BYTE *)(v9 + 8) = 5;
  *(_QWORD *)v9 = 0;
  *(_QWORD *)(v9 + 12) = v10 & 0xE0000000000000LL | 0x30A020000000001LL;
  qword_4D03D40[5] = v9;
  v11 = sub_823970(24);
  v12 = *(_QWORD *)(v11 + 12);
  *(_BYTE *)(v11 + 8) = 6;
  *(_QWORD *)(v11 + 12) = v12 & 0xE0000000000000LL | 0x80DA12B00000001LL;
  *(_QWORD *)v11 = unk_4D03E90;
  qword_4D03D40[6] = v11;
  unk_4D03E90 = v11;
  v13 = sub_823970(24);
  v14 = *(_QWORD *)(v13 + 12);
  *(_BYTE *)(v13 + 8) = 7;
  *(_QWORD *)(v13 + 12) = v14 & 0xE0000000000000LL | 0x80DA12C00000001LL;
  *(_QWORD *)v13 = unk_4D03E90;
  qword_4D03D40[7] = v13;
  unk_4D03E90 = v13;
  v15 = sub_823970(24);
  v16 = *(_QWORD *)(v15 + 12);
  *(_BYTE *)(v15 + 8) = 8;
  *(_QWORD *)(v15 + 12) = v16 & 0xE0000000000000LL | 0x80C782A00000003LL;
  *(_QWORD *)v15 = unk_4D03E90;
  qword_4D03D40[8] = v15;
  unk_4D03E90 = v15;
  v17 = sub_823970(24);
  v18 = *(_QWORD *)(v17 + 12);
  *(_BYTE *)(v17 + 8) = 9;
  *(_QWORD *)(v17 + 12) = v18 & 0xE0000000000000LL | 0x509722D00000001LL;
  *(_QWORD *)v17 = unk_4D03E90;
  qword_4D03D40[9] = v17;
  unk_4D03E90 = v17;
  v19 = sub_823970(24);
  v20 = *(_QWORD *)(v19 + 12);
  *(_BYTE *)(v19 + 8) = 10;
  *(_QWORD *)(v19 + 12) = v20 & 0xE0000000000000LL | 0x809632E00000001LL;
  *(_QWORD *)v19 = unk_4D03E90;
  qword_4D03D40[10] = v19;
  unk_4D03E90 = v19;
  v21 = sub_823970(24);
  v22 = *(_QWORD *)(v21 + 12);
  *(_BYTE *)(v21 + 8) = 11;
  *(_QWORD *)v21 = 0;
  *(_QWORD *)(v21 + 12) = v22 & 0xE0000000000000LL | 0x30A012F00000001LL;
  qword_4D03D40[11] = v21;
  v23 = sub_823970(24);
  v24 = *(_QWORD *)(v23 + 12);
  *(_BYTE *)(v23 + 8) = 12;
  *(_QWORD *)v23 = 0;
  *(_QWORD *)(v23 + 12) = v24 & 0xE0000000000000LL | 0x30A013000000001LL;
  qword_4D03D40[12] = v23;
  v25 = sub_823970(24);
  v26 = *(_QWORD *)(v25 + 12);
  *(_BYTE *)(v25 + 8) = 13;
  *(_QWORD *)v25 = 0;
  *(_QWORD *)(v25 + 12) = v26 & 0xE0000000000000LL | 0x30A013100000001LL;
  qword_4D03D40[13] = v25;
  v27 = sub_823970(24);
  v28 = *(_QWORD *)(v27 + 12);
  *(_BYTE *)(v27 + 8) = 14;
  *(_QWORD *)(v27 + 12) = v28 & 0xE0000000000000LL | 0x30C583200000002LL;
  *(_QWORD *)v27 = unk_4D03E90;
  v29 = dword_4F077C4 == 2;
  unk_4D03E90 = v27;
  qword_4D03D40[14] = v27;
  if ( v29 )
  {
    v63 = sub_823970(24);
    v64 = *(_QWORD *)(v63 + 12);
    *(_BYTE *)(v63 + 8) = 15;
    *(_QWORD *)(v63 + 12) = v64 & 0xE0000000000000LL | 0x808601100000002LL;
    *(_QWORD *)v63 = unk_4D03E90;
    qword_4D03D40[15] = v63;
    unk_4D03E90 = v63;
    v65 = sub_823970(24);
    v66 = *(_QWORD *)(v65 + 12);
    *(_BYTE *)(v65 + 8) = 16;
    *(_QWORD *)(v65 + 12) = v66 & 0xE0000000000000LL | 0x808601100000002LL;
    *(_QWORD *)v65 = unk_4D03E90;
    qword_4D03D40[16] = v65;
    unk_4D03E90 = v65;
    v67 = sub_823970(24);
    v68 = unk_4D03E90;
    v69 = *(_QWORD *)(v67 + 12) & 0xE0000000000000LL;
    *(_BYTE *)(v67 + 8) = 17;
    *(_QWORD *)v67 = v68;
    *(_QWORD *)(v67 + 12) = v69 | 0x808601100000002LL;
    unk_4D03E90 = v67;
    qword_4D03D40[17] = v67;
  }
  if ( unk_4D043EC )
  {
    v61 = sub_823970(24);
    v62 = *(_QWORD *)(v61 + 12) & 0xE0000000000000LL;
    *(_BYTE *)(v61 + 8) = 19;
    *(_QWORD *)(v61 + 12) = v62 | 0x808201200000002LL;
    *(_QWORD *)v61 = unk_4D03E90;
    unk_4D03E90 = v61;
    qword_4D03D40[19] = v61;
  }
  v30 = sub_823970(24);
  v31 = *(_QWORD *)(v30 + 12);
  *(_BYTE *)(v30 + 8) = 20;
  *(_QWORD *)(v30 + 12) = v31 & 0xE0000000000000LL | 0x808B41300000002LL;
  *(_QWORD *)v30 = unk_4D03E90;
  qword_4D03D40[20] = v30;
  unk_4D03E90 = v30;
  v32 = sub_823970(24);
  v33 = *(_QWORD *)(v32 + 12);
  *(_BYTE *)(v32 + 8) = 21;
  *(_QWORD *)(v32 + 12) = v33 & 0xE0000000000000LL | 0x808241400000002LL;
  *(_QWORD *)v32 = unk_4D03E90;
  unk_4D03E90 = v32;
  qword_4D03D40[21] = v32;
  sub_853D40(0x16u, 21, 1, 0, 0, 0, 1);
  sub_853D40(0x17u, 22, 1, 0, 0, 0, 0);
  sub_853D40(0x18u, 22, 1, 0, 0, 0, 0);
  if ( dword_4F077C4 != 2 )
    goto LABEL_6;
  v58 = sub_823970(24);
  v59 = *(_QWORD *)(v58 + 12) & 0xE0000000000000LL;
  *(_BYTE *)(v58 + 8) = 25;
  v60 = unk_4D03E90;
  *(_QWORD *)(v58 + 12) = v59 | 0x809111700000001LL;
  *(_QWORD *)v58 = v60;
  v29 = dword_4F077C4 == 2;
  unk_4D03E90 = v58;
  qword_4D03D40[25] = v58;
  if ( v29 )
  {
    if ( unk_4F07778 <= 201102 && !dword_4F07774 )
    {
LABEL_19:
      if ( !unk_4D04778 )
        goto LABEL_8;
    }
  }
  else
  {
LABEL_6:
    if ( unk_4F07778 <= 199900 )
      goto LABEL_19;
  }
  v34 = sub_823970(24);
  v35 = *(_QWORD *)(v34 + 12) & 0xE0000000000000LL;
  *(_BYTE *)(v34 + 8) = 26;
  *(_QWORD *)(v34 + 12) = v35 | 0x80C001800000002LL;
  *(_QWORD *)v34 = unk_4D03E90;
  unk_4D03E90 = v34;
  qword_4D03D40[26] = v34;
LABEL_8:
  v36 = sub_823970(24);
  v37 = *(_QWORD *)(v36 + 12);
  *(_BYTE *)(v36 + 8) = 27;
  *(_QWORD *)(v36 + 12) = v37 & 0xE0000000000000LL | 0x80C041900000002LL;
  *(_QWORD *)v36 = unk_4D03E90;
  v38 = HIDWORD(qword_4F077B4);
  unk_4D03E90 = v36;
  qword_4D03D40[27] = v36;
  if ( v38 && qword_4F077A8 > 0x9D07u )
  {
    v70 = sub_823970(24);
    v71 = *(_QWORD *)(v70 + 12);
    *(_BYTE *)(v70 + 8) = 29;
    *(_QWORD *)(v70 + 12) = v71 & 0xE0000000000000LL | 0x80C101A00000002LL;
    *(_QWORD *)v70 = unk_4D03E90;
    qword_4D03D40[29] = v70;
    unk_4D03E90 = v70;
    v72 = sub_823970(24);
    v73 = *(_QWORD *)(v72 + 12) & 0xE0000000000000LL;
    *(_BYTE *)(v72 + 8) = 28;
    v74 = unk_4D03E90;
    *(_QWORD *)(v72 + 12) = v73 | 0x80C501A00000003LL;
    *(_QWORD *)v72 = v74;
    unk_4D03E90 = v72;
    qword_4D03D40[28] = v72;
  }
  v39 = sub_823970(24);
  v40 = *(_QWORD *)(v39 + 12);
  *(_BYTE *)(v39 + 8) = 30;
  *(_QWORD *)(v39 + 12) = v40 & 0xE0000000000000LL | 0x809001B00000003LL;
  *(_QWORD *)v39 = unk_4D03E90;
  qword_4D03D40[30] = v39;
  unk_4D03E90 = v39;
  v41 = sub_823970(24);
  v42 = *(_QWORD *)(v41 + 12);
  *(_BYTE *)(v41 + 8) = 31;
  *(_QWORD *)(v41 + 12) = v42 & 0xE0000000000000LL | 0x809001B00000003LL;
  *(_QWORD *)v41 = unk_4D03E90;
  qword_4D03D40[31] = v41;
  unk_4D03E90 = v41;
  v43 = sub_823970(24);
  v44 = *(_QWORD *)(v43 + 12);
  *(_BYTE *)(v43 + 8) = 32;
  *(_QWORD *)(v43 + 12) = v44 & 0xE0000000000000LL | 0x809001B00000003LL;
  *(_QWORD *)v43 = unk_4D03E90;
  qword_4D03D40[32] = v43;
  unk_4D03E90 = v43;
  v45 = sub_823970(24);
  v46 = *(_QWORD *)(v45 + 12);
  *(_BYTE *)(v45 + 8) = 33;
  *(_QWORD *)(v45 + 12) = v46 & 0xE0000000000000LL | 0x809001B00000003LL;
  *(_QWORD *)v45 = unk_4D03E90;
  qword_4D03D40[33] = v45;
  unk_4D03E90 = v45;
  v47 = sub_823970(24);
  v48 = *(_QWORD *)(v47 + 12);
  *(_BYTE *)(v47 + 8) = 34;
  *(_QWORD *)(v47 + 12) = v48 & 0xE0000000000000LL | 0x809001B00000003LL;
  *(_QWORD *)v47 = unk_4D03E90;
  qword_4D03D40[34] = v47;
  unk_4D03E90 = v47;
  v49 = sub_823970(24);
  v50 = *(_QWORD *)(v49 + 12);
  *(_BYTE *)(v49 + 8) = 35;
  v51 = v50 & 0xE0000000000000LL | 0x809001B00000003LL;
  v52 = unk_4D03E90;
  *(_QWORD *)(v49 + 12) = v51;
  *(_QWORD *)v49 = v52;
  qword_4D03D40[35] = v49;
  unk_4D03E90 = v49;
  v53 = sub_823970(24);
  v54 = *(_QWORD *)(v53 + 12) & 0xE0000000000000LL;
  *(_BYTE *)(v53 + 8) = 36;
  v55 = unk_4D03E90;
  *(_QWORD *)(v53 + 12) = v54 | 0x809001C00000003LL;
  *(_QWORD *)v53 = v55;
  unk_4D03E90 = v53;
  qword_4D03D40[36] = v53;
  if ( (_DWORD)qword_4F077B4 || HIDWORD(qword_4F077B4) && qword_4F077A8 > 0x9DD2u )
  {
    sub_853D40(0x25u, 29, 0, 1, 1, 1, 1);
    sub_853D40(0x26u, 30, 0, 1, 1, 1, 1);
  }
  result = sub_823970(24);
  v57 = *(_QWORD *)(result + 12) & 0xE0000000000000LL;
  *(_BYTE *)(result + 8) = 39;
  *(_QWORD *)(result + 12) = v57 | 0x508980000000002LL;
  *(_QWORD *)result = unk_4D03E90;
  unk_4D03E90 = result;
  qword_4D03D40[39] = result;
  return result;
}
