// Function: ctor_297_0
// Address: 0x4fce80
//
int __fastcall ctor_297_0(__int64 a1, int a2, int a3, __int64 a4, int a5, int a6)
{
  int v6; // eax
  const char *v7; // rsi
  _BYTE *v8; // r14
  _BYTE *v9; // rbx
  unsigned int v10; // esi
  const char *v11; // r8
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r10
  __int64 v15; // r9
  __int64 v16; // rax
  int v17; // r8d
  int v18; // r9d
  int v19; // eax
  const char *v20; // rsi
  __int64 v21; // rcx
  _BYTE *v22; // r14
  _BYTE *v23; // r15
  const char *v24; // r8
  __int64 v25; // rdx
  int v26; // r9d
  __int64 v27; // r11
  __int64 v28; // r10
  unsigned int v29; // esi
  __int64 v30; // rax
  int v31; // eax
  __int128 v33; // [rsp-A0h] [rbp-330h]
  __int128 v34; // [rsp-A0h] [rbp-330h]
  __int128 v35; // [rsp-90h] [rbp-320h]
  __int128 v36; // [rsp-90h] [rbp-320h]
  __int128 v37; // [rsp-78h] [rbp-308h]
  __int128 v38; // [rsp-78h] [rbp-308h]
  __int128 v39; // [rsp-68h] [rbp-2F8h]
  __int128 v40; // [rsp-68h] [rbp-2F8h]
  __int128 v41; // [rsp-50h] [rbp-2E0h]
  __int128 v42; // [rsp-50h] [rbp-2E0h]
  __int128 v43; // [rsp-40h] [rbp-2D0h]
  __int128 v44; // [rsp-40h] [rbp-2D0h]
  __int128 v45; // [rsp-28h] [rbp-2B8h]
  __int128 v46; // [rsp-28h] [rbp-2B8h]
  __int128 v47; // [rsp-18h] [rbp-2A8h]
  __int128 v48; // [rsp-18h] [rbp-2A8h]
  __int64 v49; // [rsp+0h] [rbp-290h]
  __int64 v50; // [rsp+0h] [rbp-290h]
  int v51; // [rsp+8h] [rbp-288h]
  const char *v52; // [rsp+8h] [rbp-288h]
  const char *v53; // [rsp+10h] [rbp-280h]
  __int64 v54; // [rsp+10h] [rbp-280h]
  __int64 v55; // [rsp+18h] [rbp-278h]
  unsigned int v56; // [rsp+18h] [rbp-278h]
  __int64 v57; // [rsp+20h] [rbp-270h]
  __int64 v58; // [rsp+20h] [rbp-270h]
  __int64 v59; // [rsp+40h] [rbp-250h]
  __int64 v60; // [rsp+70h] [rbp-220h]
  __int64 v61; // [rsp+A0h] [rbp-1F0h]
  __int64 v62; // [rsp+D0h] [rbp-1C0h]
  __int64 v63; // [rsp+100h] [rbp-190h]
  __int64 v64; // [rsp+130h] [rbp-160h]
  __int64 v65; // [rsp+160h] [rbp-130h]
  __int64 v66; // [rsp+190h] [rbp-100h]
  _BYTE *v67; // [rsp+1B0h] [rbp-E0h] BYREF
  int v68; // [rsp+1B8h] [rbp-D8h]
  _BYTE v69[208]; // [rsp+1C0h] [rbp-D0h] BYREF

  *((_QWORD *)&v47 + 1) = "display a graph using the real profile count if available.";
  LODWORD(v66) = 3;
  LODWORD(v65) = 2;
  LODWORD(v64) = 1;
  LODWORD(v63) = 0;
  *(_QWORD *)&v47 = v66;
  *((_QWORD *)&v45 + 1) = 5;
  *(_QWORD *)&v45 = "count";
  *((_QWORD *)&v43 + 1) = "display a graph using the raw integer fractional block frequency representation.";
  *(_QWORD *)&v43 = v65;
  *((_QWORD *)&v41 + 1) = 7;
  *(_QWORD *)&v41 = "integer";
  *((_QWORD *)&v39 + 1) = "display a graph using the fractional block frequency representation.";
  *(_QWORD *)&v39 = v64;
  *((_QWORD *)&v37 + 1) = 8;
  *(_QWORD *)&v37 = "fraction";
  *((_QWORD *)&v35 + 1) = "do not display graphs.";
  *(_QWORD *)&v35 = v63;
  *((_QWORD *)&v33 + 1) = 4;
  *(_QWORD *)&v33 = "none";
  sub_12F1570(
    (unsigned int)&v67,
    a2,
    a3,
    (unsigned int)"fraction",
    a5,
    a6,
    v33,
    v35,
    22,
    v37,
    v39,
    68,
    v41,
    v43,
    80,
    v45,
    v47,
    58);
  qword_4FC48A0 = (__int64)&unk_49EED30;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  v7 = "view-machine-block-freq-propagation-dags";
  word_4FC48AC &= 0xF000u;
  qword_4FC48B0 = 0;
  qword_4FC48B8 = 0;
  qword_4FC48C0 = 0;
  qword_4FC48C8 = 0;
  dword_4FC48A8 = v6;
  qword_4FC48D0 = 0;
  qword_4FC48E8 = (__int64)qword_4FA01C0;
  qword_4FC48F8 = (__int64)&unk_4FC4918;
  qword_4FC4900 = (__int64)&unk_4FC4918;
  qword_4FC48D8 = 0;
  qword_4FC48E0 = 0;
  qword_4FC48A0 = (__int64)&unk_49E8808;
  qword_4FC48F0 = 0;
  byte_4FC4938 = 0;
  qword_4FC4958 = (__int64)&unk_49E87B8;
  qword_4FC4968 = (__int64)&unk_4FC4978;
  qword_4FC4970 = 0x800000000LL;
  qword_4FC4908 = 4;
  dword_4FC4910 = 0;
  dword_4FC4940 = 0;
  qword_4FC4948 = (__int64)&unk_49E8798;
  byte_4FC4954 = 1;
  dword_4FC4950 = 0;
  qword_4FC4960 = (__int64)&qword_4FC48A0;
  sub_16B8280(&qword_4FC48A0, "view-machine-block-freq-propagation-dags", 40);
  qword_4FC48D0 = 97;
  v8 = v67;
  LOBYTE(word_4FC48AC) = word_4FC48AC & 0x9F | 0x20;
  qword_4FC48C8 = (__int64)"Pop up a window to show a dag displaying how machine block frequencies propagate through the CFG.";
  v9 = &v67[40 * v68];
  if ( v67 != v9 )
  {
    do
    {
      v10 = qword_4FC4970;
      v11 = *(const char **)v8;
      v12 = *((_QWORD *)v8 + 1);
      v13 = *((unsigned int *)v8 + 4);
      v14 = *((_QWORD *)v8 + 3);
      v15 = *((_QWORD *)v8 + 4);
      if ( (unsigned int)qword_4FC4970 >= HIDWORD(qword_4FC4970) )
      {
        v50 = *((_QWORD *)v8 + 4);
        v52 = *(const char **)v8;
        v54 = *((_QWORD *)v8 + 1);
        v56 = *((_DWORD *)v8 + 4);
        v58 = *((_QWORD *)v8 + 3);
        sub_1DE0770(&qword_4FC4968, 0);
        v10 = qword_4FC4970;
        v15 = v50;
        v11 = v52;
        v12 = v54;
        v13 = v56;
        v14 = v58;
      }
      v16 = qword_4FC4968 + 48LL * v10;
      if ( v16 )
      {
        *(_QWORD *)v16 = v11;
        *(_QWORD *)(v16 + 8) = v12;
        *(_QWORD *)(v16 + 16) = v14;
        *(_QWORD *)(v16 + 24) = v15;
        *(_DWORD *)(v16 + 40) = v13;
        *(_BYTE *)(v16 + 44) = 1;
        *(_QWORD *)(v16 + 32) = &unk_49E8798;
        v10 = qword_4FC4970;
      }
      v8 += 40;
      LODWORD(qword_4FC4970) = v10 + 1;
      v7 = v11;
      sub_16B7FD0(qword_4FC4960, v11, v12, v13);
    }
    while ( v9 != v8 );
  }
  sub_16B88A0(&qword_4FC48A0);
  if ( v67 != v69 )
    _libc_free(v67, v7);
  __cxa_atexit(sub_13678E0, &qword_4FC48A0, &qword_4A427C0);
  LODWORD(v62) = 3;
  *((_QWORD *)&v48 + 1) = "display a graph using the real profile count if available.";
  *(_QWORD *)&v48 = v62;
  *((_QWORD *)&v46 + 1) = 5;
  *(_QWORD *)&v46 = "count";
  LODWORD(v61) = 2;
  LODWORD(v60) = 1;
  LODWORD(v59) = 0;
  *((_QWORD *)&v44 + 1) = "display a graph using the raw integer fractional block frequency representation.";
  *(_QWORD *)&v44 = v61;
  *((_QWORD *)&v42 + 1) = 7;
  *(_QWORD *)&v42 = "integer";
  *((_QWORD *)&v40 + 1) = "display a graph using the fractional block frequency representation.";
  *(_QWORD *)&v40 = v60;
  *((_QWORD *)&v38 + 1) = 8;
  *(_QWORD *)&v38 = "fraction";
  *((_QWORD *)&v36 + 1) = "do not display graphs.";
  *(_QWORD *)&v36 = v59;
  *((_QWORD *)&v34 + 1) = 4;
  *(_QWORD *)&v34 = "none";
  sub_12F1570(
    (unsigned int)&v67,
    (unsigned int)&qword_4FC48A0,
    (unsigned int)"do not display graphs.",
    (unsigned int)"fraction",
    v17,
    v18,
    v34,
    v36,
    22,
    v38,
    v40,
    68,
    v42,
    v44,
    80,
    v46,
    v48,
    58);
  qword_4FC4640[0] = &unk_49EED30;
  v19 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  v20 = "view-block-layout-with-bfi";
  WORD2(qword_4FC4640[1]) &= 0xF000u;
  LODWORD(qword_4FC4640[1]) = v19;
  qword_4FC4640[2] = 0;
  qword_4FC4640[9] = qword_4FA01C0;
  qword_4FC4640[11] = &qword_4FC4640[15];
  qword_4FC4640[12] = &qword_4FC4640[15];
  qword_4FC4640[21] = &unk_49E8798;
  qword_4FC4640[3] = 0;
  qword_4FC4640[4] = 0;
  qword_4FC4640[0] = &unk_49E8808;
  qword_4FC4640[5] = 0;
  qword_4FC4640[6] = 0;
  qword_4FC4640[23] = &unk_49E87B8;
  qword_4FC4640[25] = &qword_4FC4640[27];
  qword_4FC4640[26] = 0x800000000LL;
  qword_4FC4640[7] = 0;
  qword_4FC4640[8] = 0;
  qword_4FC4640[10] = 0;
  qword_4FC4640[13] = 4;
  LODWORD(qword_4FC4640[14]) = 0;
  LOBYTE(qword_4FC4640[19]) = 0;
  LODWORD(qword_4FC4640[20]) = 0;
  BYTE4(qword_4FC4640[22]) = 1;
  LODWORD(qword_4FC4640[22]) = 0;
  qword_4FC4640[24] = qword_4FC4640;
  sub_16B8280(qword_4FC4640, "view-block-layout-with-bfi", 26);
  qword_4FC4640[6] = 96;
  BYTE4(qword_4FC4640[1]) = BYTE4(qword_4FC4640[1]) & 0x9F | 0x20;
  qword_4FC4640[5] = "Pop up a window to show a dag displaying MBP layout and associated block frequencies of the CFG.";
  v22 = &v67[40 * v68];
  v23 = v67;
  while ( v22 != v23 )
  {
    v24 = *(const char **)v23;
    v25 = *((_QWORD *)v23 + 1);
    v26 = *((_DWORD *)v23 + 4);
    v27 = *((_QWORD *)v23 + 3);
    v28 = *((_QWORD *)v23 + 4);
    v29 = qword_4FC4640[26];
    if ( LODWORD(qword_4FC4640[26]) >= HIDWORD(qword_4FC4640[26]) )
    {
      v49 = *((_QWORD *)v23 + 1);
      v51 = *((_DWORD *)v23 + 4);
      v53 = *(const char **)v23;
      v55 = *((_QWORD *)v23 + 3);
      v57 = *((_QWORD *)v23 + 4);
      sub_1DE0770(&qword_4FC4640[25], 0);
      v29 = qword_4FC4640[26];
      v25 = v49;
      v26 = v51;
      v24 = v53;
      v27 = v55;
      v28 = v57;
    }
    v30 = qword_4FC4640[25] + 48LL * v29;
    if ( v30 )
    {
      *(_QWORD *)v30 = v24;
      *(_QWORD *)(v30 + 8) = v25;
      *(_QWORD *)(v30 + 16) = v27;
      *(_QWORD *)(v30 + 24) = v28;
      *(_DWORD *)(v30 + 40) = v26;
      *(_BYTE *)(v30 + 44) = 1;
      *(_QWORD *)(v30 + 32) = &unk_49E8798;
      v29 = qword_4FC4640[26];
    }
    v23 += 40;
    LODWORD(qword_4FC4640[26]) = v29 + 1;
    v20 = v24;
    sub_16B7FD0(qword_4FC4640[24], v24, v25, v21);
  }
  sub_16B88A0(qword_4FC4640);
  if ( v67 != v69 )
    _libc_free(v67, v20);
  __cxa_atexit(sub_13678E0, qword_4FC4640, &qword_4A427C0);
  qword_4FC4560 = (__int64)&unk_49EED30;
  v31 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC456C &= 0xF000u;
  qword_4FC4570 = 0;
  qword_4FC4578 = 0;
  qword_4FC4580 = 0;
  qword_4FC4588 = 0;
  dword_4FC4568 = v31;
  qword_4FC4590 = 0;
  qword_4FC45A8 = (__int64)qword_4FA01C0;
  qword_4FC45B8 = (__int64)&unk_4FC45D8;
  qword_4FC45C0 = (__int64)&unk_4FC45D8;
  qword_4FC4598 = 0;
  qword_4FC45A0 = 0;
  qword_4FC4608 = (__int64)&unk_49E74E8;
  word_4FC4610 = 256;
  qword_4FC45B0 = 0;
  byte_4FC45F8 = 0;
  qword_4FC4560 = (__int64)&unk_49EEC70;
  qword_4FC45C8 = 4;
  byte_4FC4600 = 0;
  qword_4FC4618 = (__int64)&unk_49EEDB0;
  dword_4FC45D0 = 0;
  sub_16B8280(&qword_4FC4560, "print-machine-bfi", 17);
  word_4FC4610 = 256;
  byte_4FC4600 = 0;
  qword_4FC4590 = 39;
  LOBYTE(word_4FC456C) = word_4FC456C & 0x9F | 0x20;
  qword_4FC4588 = (__int64)"Print the machine block frequency info.";
  sub_16B88A0(&qword_4FC4560);
  return __cxa_atexit(sub_12EDEC0, &qword_4FC4560, &qword_4A427C0);
}
