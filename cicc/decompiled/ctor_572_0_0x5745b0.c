// Function: ctor_572_0
// Address: 0x5745b0
//
int __fastcall ctor_572_0(__int64 a1, int a2, int a3, int a4, int a5, int a6)
{
  int v6; // edx
  int v7; // ecx
  int v8; // r8d
  int v9; // r9d
  int v10; // edx
  __int64 v11; // r13
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // edx
  __int64 v18; // rax
  __int64 v19; // rdx
  int v20; // edx
  __int64 v21; // rax
  __int64 v22; // rdx
  int v23; // edx
  __int64 v24; // rax
  __int64 v25; // rdx
  int v26; // edx
  __int64 v27; // rax
  __int64 v28; // rdx
  int v29; // edx
  __int64 v30; // rax
  __int64 v31; // rdx
  int v32; // edx
  __int64 v33; // rax
  __int64 v34; // rdx
  int v35; // edx
  __int64 v36; // r12
  __int64 v37; // rax
  unsigned __int64 v38; // rdx
  __int128 v40; // [rsp-78h] [rbp-2C0h]
  __int128 v41; // [rsp-78h] [rbp-2C0h]
  __int128 v42; // [rsp-68h] [rbp-2B0h]
  __int128 v43; // [rsp-68h] [rbp-2B0h]
  __int128 v44; // [rsp-50h] [rbp-298h]
  __int128 v45; // [rsp-50h] [rbp-298h]
  __int128 v46; // [rsp-40h] [rbp-288h]
  __int128 v47; // [rsp-40h] [rbp-288h]
  __int128 v48; // [rsp-28h] [rbp-270h]
  __int128 v49; // [rsp-28h] [rbp-270h]
  __int128 v50; // [rsp-18h] [rbp-260h]
  __int128 v51; // [rsp-18h] [rbp-260h]
  __int64 v52; // [rsp+10h] [rbp-238h]
  __int64 v53; // [rsp+10h] [rbp-238h]
  __int64 v54; // [rsp+10h] [rbp-238h]
  __int64 v55; // [rsp+10h] [rbp-238h]
  __int64 v56; // [rsp+10h] [rbp-238h]
  __int64 v57; // [rsp+10h] [rbp-238h]
  __int64 v58; // [rsp+20h] [rbp-228h]
  int v59; // [rsp+28h] [rbp-220h] BYREF
  int v60; // [rsp+2Ch] [rbp-21Ch] BYREF
  int *v61; // [rsp+30h] [rbp-218h] BYREF
  _QWORD v62[2]; // [rsp+38h] [rbp-210h] BYREF
  _QWORD v63[2]; // [rsp+48h] [rbp-200h] BYREF
  __int64 v64; // [rsp+58h] [rbp-1F0h]
  const char *v65; // [rsp+60h] [rbp-1E8h]
  __int64 v66; // [rsp+68h] [rbp-1E0h]
  _QWORD v67[2]; // [rsp+78h] [rbp-1D0h] BYREF
  __int64 v68; // [rsp+88h] [rbp-1C0h]
  const char *v69; // [rsp+90h] [rbp-1B8h]
  __int64 v70; // [rsp+98h] [rbp-1B0h]
  const char *v71; // [rsp+A8h] [rbp-1A0h] BYREF
  __int64 v72; // [rsp+B0h] [rbp-198h]
  __int64 v73; // [rsp+B8h] [rbp-190h]
  const char *v74; // [rsp+C0h] [rbp-188h]
  __int64 v75; // [rsp+C8h] [rbp-180h]
  char *v76; // [rsp+D8h] [rbp-170h]
  __int64 v77; // [rsp+E0h] [rbp-168h]
  __int64 v78; // [rsp+E8h] [rbp-160h]
  const char *v79; // [rsp+F0h] [rbp-158h]
  __int64 v80; // [rsp+F8h] [rbp-150h]
  char *v81; // [rsp+108h] [rbp-140h]
  __int64 v82; // [rsp+110h] [rbp-138h]
  __int64 v83; // [rsp+118h] [rbp-130h]
  const char *v84; // [rsp+120h] [rbp-128h]
  __int64 v85; // [rsp+128h] [rbp-120h]
  const char *v86; // [rsp+138h] [rbp-110h]
  __int64 v87; // [rsp+140h] [rbp-108h]
  __int64 v88; // [rsp+148h] [rbp-100h]
  const char *v89; // [rsp+150h] [rbp-F8h]
  __int64 v90; // [rsp+158h] [rbp-F0h]
  const char *v91; // [rsp+168h] [rbp-E0h] BYREF
  __int64 v92; // [rsp+170h] [rbp-D8h]
  _BYTE v93[208]; // [rsp+178h] [rbp-D0h] BYREF

  v89 = "Force bidirectional pre reg-alloc list scheduling";
  v84 = "Force bottom-up pre reg-alloc list scheduling";
  v79 = "Force top-down pre reg-alloc list scheduling";
  v87 = 13;
  LODWORD(v88) = 3;
  v90 = 49;
  v86 = "bidirectional";
  v81 = "bottomup";
  *((_QWORD *)&v50 + 1) = "Force bidirectional pre reg-alloc list scheduling";
  v82 = 8;
  *(_QWORD *)&v50 = v88;
  LODWORD(v83) = 2;
  *((_QWORD *)&v48 + 1) = 13;
  v85 = 45;
  *(_QWORD *)&v48 = "bidirectional";
  v76 = "topdown";
  v77 = 7;
  LODWORD(v78) = 1;
  v80 = 44;
  *((_QWORD *)&v46 + 1) = "Force bottom-up pre reg-alloc list scheduling";
  *(_QWORD *)&v46 = v83;
  *((_QWORD *)&v44 + 1) = 8;
  *(_QWORD *)&v44 = "bottomup";
  *((_QWORD *)&v42 + 1) = "Force top-down pre reg-alloc list scheduling";
  *(_QWORD *)&v42 = v78;
  *((_QWORD *)&v40 + 1) = 7;
  *(_QWORD *)&v40 = "topdown";
  sub_22735E0((unsigned int)&v91, a2, a3, a4, a5, a6, v40, v42, 44, v44, v46, 45, v48, v50, 49);
  v67[0] = v63;
  v71 = "Pre reg-alloc list scheduling direction";
  LODWORD(v63[0]) = 0;
  v72 = 39;
  LODWORD(v62[0]) = 1;
  sub_2ED0A80(&unk_5021AC0, "misched-prera-direction", v62, &v71, v67, &v91);
  if ( v91 != v93 )
    _libc_free(v91, "misched-prera-direction");
  __cxa_atexit(sub_2EC2240, &unk_5021AC0, &qword_4A427C0);
  v72 = 13;
  v74 = "Force bidirectional post reg-alloc list scheduling";
  v69 = "Force bottom-up post reg-alloc list scheduling";
  v65 = "Force top-down post reg-alloc list scheduling";
  LODWORD(v73) = 3;
  v75 = 50;
  v67[1] = 8;
  LODWORD(v68) = 2;
  *((_QWORD *)&v51 + 1) = "Force bidirectional post reg-alloc list scheduling";
  v70 = 46;
  *(_QWORD *)&v51 = v73;
  v63[0] = "topdown";
  *((_QWORD *)&v49 + 1) = 13;
  v71 = "bidirectional";
  *(_QWORD *)&v49 = "bidirectional";
  v67[0] = "bottomup";
  v63[1] = 7;
  *((_QWORD *)&v47 + 1) = "Force bottom-up post reg-alloc list scheduling";
  LODWORD(v64) = 1;
  *(_QWORD *)&v47 = v68;
  v66 = 45;
  *((_QWORD *)&v45 + 1) = 8;
  *(_QWORD *)&v45 = "bottomup";
  *((_QWORD *)&v43 + 1) = "Force top-down post reg-alloc list scheduling";
  *(_QWORD *)&v43 = v64;
  *((_QWORD *)&v41 + 1) = 7;
  *(_QWORD *)&v41 = "topdown";
  sub_22735E0((unsigned int)&v91, (unsigned int)&unk_5021AC0, v6, v7, v8, v9, v41, v43, 45, v45, v47, 46, v49, v51, 50);
  v61 = &v60;
  v60 = 0;
  v62[0] = "Post reg-alloc list scheduling direction";
  v62[1] = 40;
  v59 = 1;
  sub_2ED0E10(&unk_5021860, "misched-postra-direction", &v59, v62, &v61, &v91);
  if ( v91 != v93 )
    _libc_free(v91, "misched-postra-direction");
  __cxa_atexit(sub_2EC2240, &unk_5021860, &qword_4A427C0);
  qword_5021780 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_502178C &= 0x8000u;
  word_5021790 = 0;
  qword_50217D0 = 0x100000000LL;
  qword_5021798 = 0;
  qword_50217A0 = 0;
  qword_50217A8 = 0;
  dword_5021788 = v10;
  qword_50217B0 = 0;
  qword_50217B8 = 0;
  qword_50217C0 = 0;
  qword_50217C8 = (__int64)&unk_50217D8;
  qword_50217E0 = 0;
  qword_50217E8 = (__int64)&unk_5021800;
  qword_50217F0 = 1;
  dword_50217F8 = 0;
  byte_50217FC = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_50217D0;
  v13 = (unsigned int)qword_50217D0 + 1LL;
  if ( v13 > HIDWORD(qword_50217D0) )
  {
    sub_C8D5F0((char *)&unk_50217D8 - 16, &unk_50217D8, v13, 8);
    v12 = (unsigned int)qword_50217D0;
  }
  *(_QWORD *)(qword_50217C8 + 8 * v12) = v11;
  LODWORD(qword_50217D0) = qword_50217D0 + 1;
  qword_5021808 = 0;
  qword_5021810 = (__int64)&unk_49D9748;
  qword_5021818 = 0;
  qword_5021780 = (__int64)&unk_49DC090;
  qword_5021820 = (__int64)&unk_49DC1D0;
  qword_5021840 = (__int64)nullsub_23;
  qword_5021838 = (__int64)sub_984030;
  sub_C53080(&qword_5021780, "misched-dcpl", 12);
  qword_50217B0 = 36;
  LOBYTE(dword_502178C) = dword_502178C & 0x9F | 0x20;
  qword_50217A8 = (__int64)"Print critical path length to stdout";
  sub_C53130(&qword_5021780);
  __cxa_atexit(sub_984900, &qword_5021780, &qword_4A427C0);
  qword_50216A0 = &unk_49DC150;
  v14 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_50216AC = word_50216AC & 0x8000;
  unk_50216B0 = 0;
  qword_50216E8[1] = 0x100000000LL;
  unk_50216A8 = v14;
  qword_50216E8[0] = &qword_50216E8[2];
  unk_50216B8 = 0;
  unk_50216C0 = 0;
  unk_50216C8 = 0;
  unk_50216D0 = 0;
  unk_50216D8 = 0;
  unk_50216E0 = 0;
  qword_50216E8[3] = 0;
  qword_50216E8[4] = &qword_50216E8[7];
  qword_50216E8[5] = 1;
  LODWORD(qword_50216E8[6]) = 0;
  BYTE4(qword_50216E8[6]) = 1;
  v15 = sub_C57470();
  v16 = LODWORD(qword_50216E8[1]);
  if ( (unsigned __int64)LODWORD(qword_50216E8[1]) + 1 > HIDWORD(qword_50216E8[1]) )
  {
    v52 = v15;
    sub_C8D5F0(qword_50216E8, &qword_50216E8[2], LODWORD(qword_50216E8[1]) + 1LL, 8);
    v16 = LODWORD(qword_50216E8[1]);
    v15 = v52;
  }
  *(_QWORD *)(qword_50216E8[0] + 8 * v16) = v15;
  ++LODWORD(qword_50216E8[1]);
  qword_50216E8[8] = 0;
  qword_50216E8[9] = &unk_49D9748;
  qword_50216E8[10] = 0;
  qword_50216A0 = &unk_49DC090;
  qword_50216E8[11] = &unk_49DC1D0;
  qword_50216E8[15] = nullsub_23;
  qword_50216E8[14] = sub_984030;
  sub_C53080(&qword_50216A0, "verify-misched", 14);
  unk_50216D0 = 57;
  LOBYTE(word_50216AC) = word_50216AC & 0x9F | 0x20;
  unk_50216C8 = "Verify machine instrs before and after machine scheduling";
  sub_C53130(&qword_50216A0);
  __cxa_atexit(sub_984900, &qword_50216A0, &qword_4A427C0);
  qword_50215C0 = (__int64)&unk_49DC150;
  v17 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_50215CC &= 0x8000u;
  word_50215D0 = 0;
  qword_5021610 = 0x100000000LL;
  qword_5021608 = (__int64)&unk_5021618;
  qword_50215D8 = 0;
  qword_50215E0 = 0;
  dword_50215C8 = v17;
  qword_50215E8 = 0;
  qword_50215F0 = 0;
  qword_50215F8 = 0;
  qword_5021600 = 0;
  qword_5021620 = 0;
  qword_5021628 = (__int64)&unk_5021640;
  qword_5021630 = 1;
  dword_5021638 = 0;
  byte_502163C = 1;
  v18 = sub_C57470();
  v19 = (unsigned int)qword_5021610;
  if ( (unsigned __int64)(unsigned int)qword_5021610 + 1 > HIDWORD(qword_5021610) )
  {
    v53 = v18;
    sub_C8D5F0((char *)&unk_5021618 - 16, &unk_5021618, (unsigned int)qword_5021610 + 1LL, 8);
    v19 = (unsigned int)qword_5021610;
    v18 = v53;
  }
  *(_QWORD *)(qword_5021608 + 8 * v19) = v18;
  LODWORD(qword_5021610) = qword_5021610 + 1;
  qword_5021648 = 0;
  qword_5021650 = (__int64)&unk_49D9728;
  qword_5021658 = 0;
  qword_50215C0 = (__int64)&unk_49DBF10;
  qword_5021660 = (__int64)&unk_49DC290;
  qword_5021680 = (__int64)nullsub_24;
  qword_5021678 = (__int64)sub_984050;
  sub_C53080(&qword_50215C0, "misched-limit", 13);
  qword_50215F0 = 34;
  LODWORD(qword_5021648) = 256;
  BYTE4(qword_5021658) = 1;
  LODWORD(qword_5021658) = 256;
  LOBYTE(dword_50215CC) = dword_50215CC & 0x9F | 0x20;
  qword_50215E8 = (__int64)"Limit ready list to N instructions";
  sub_C53130(&qword_50215C0);
  __cxa_atexit(sub_984970, &qword_50215C0, &qword_4A427C0);
  qword_50214E0 = (__int64)&unk_49DC150;
  v20 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5021530 = 0x100000000LL;
  dword_50214EC &= 0x8000u;
  qword_5021528 = (__int64)&unk_5021538;
  word_50214F0 = 0;
  qword_50214F8 = 0;
  dword_50214E8 = v20;
  qword_5021500 = 0;
  qword_5021508 = 0;
  qword_5021510 = 0;
  qword_5021518 = 0;
  qword_5021520 = 0;
  qword_5021540 = 0;
  qword_5021548 = (__int64)&unk_5021560;
  qword_5021550 = 1;
  dword_5021558 = 0;
  byte_502155C = 1;
  v21 = sub_C57470();
  v22 = (unsigned int)qword_5021530;
  if ( (unsigned __int64)(unsigned int)qword_5021530 + 1 > HIDWORD(qword_5021530) )
  {
    v54 = v21;
    sub_C8D5F0((char *)&unk_5021538 - 16, &unk_5021538, (unsigned int)qword_5021530 + 1LL, 8);
    v22 = (unsigned int)qword_5021530;
    v21 = v54;
  }
  *(_QWORD *)(qword_5021528 + 8 * v22) = v21;
  LODWORD(qword_5021530) = qword_5021530 + 1;
  qword_5021568 = 0;
  qword_5021570 = (__int64)&unk_49D9748;
  qword_5021578 = 0;
  qword_50214E0 = (__int64)&unk_49DC090;
  qword_5021580 = (__int64)&unk_49DC1D0;
  qword_50215A0 = (__int64)nullsub_23;
  qword_5021598 = (__int64)sub_984030;
  sub_C53080(&qword_50214E0, "misched-regpressure", 19);
  LOWORD(qword_5021578) = 257;
  LOBYTE(qword_5021568) = 1;
  qword_5021510 = 36;
  LOBYTE(dword_50214EC) = dword_50214EC & 0x9F | 0x20;
  qword_5021508 = (__int64)"Enable register pressure scheduling.";
  sub_C53130(&qword_50214E0);
  __cxa_atexit(sub_984900, &qword_50214E0, &qword_4A427C0);
  LOBYTE(v60) = 1;
  v62[0] = &v60;
  v91 = "Enable cyclic critical path analysis.";
  v92 = 37;
  LODWORD(v61) = 1;
  sub_2ECA690(&unk_5021400, "misched-cyclicpath", &v61, &v91, v62);
  __cxa_atexit(sub_984900, &unk_5021400, &qword_4A427C0);
  qword_5021320 = (__int64)&unk_49DC150;
  v23 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5021370 = 0x100000000LL;
  dword_502132C &= 0x8000u;
  qword_5021368 = (__int64)&unk_5021378;
  word_5021330 = 0;
  qword_5021338 = 0;
  dword_5021328 = v23;
  qword_5021340 = 0;
  qword_5021348 = 0;
  qword_5021350 = 0;
  qword_5021358 = 0;
  qword_5021360 = 0;
  qword_5021380 = 0;
  qword_5021388 = (__int64)&unk_50213A0;
  qword_5021390 = 1;
  dword_5021398 = 0;
  byte_502139C = 1;
  v24 = sub_C57470();
  v25 = (unsigned int)qword_5021370;
  if ( (unsigned __int64)(unsigned int)qword_5021370 + 1 > HIDWORD(qword_5021370) )
  {
    v55 = v24;
    sub_C8D5F0((char *)&unk_5021378 - 16, &unk_5021378, (unsigned int)qword_5021370 + 1LL, 8);
    v25 = (unsigned int)qword_5021370;
    v24 = v55;
  }
  *(_QWORD *)(qword_5021368 + 8 * v25) = v24;
  LODWORD(qword_5021370) = qword_5021370 + 1;
  qword_50213A8 = 0;
  qword_50213B0 = (__int64)&unk_49D9748;
  qword_50213B8 = 0;
  qword_5021320 = (__int64)&unk_49DC090;
  qword_50213C0 = (__int64)&unk_49DC1D0;
  qword_50213E0 = (__int64)nullsub_23;
  qword_50213D8 = (__int64)sub_984030;
  sub_C53080(&qword_5021320, "misched-cluster", 15);
  LOWORD(qword_50213B8) = 257;
  LOBYTE(qword_50213A8) = 1;
  qword_5021350 = 24;
  LOBYTE(dword_502132C) = dword_502132C & 0x9F | 0x20;
  qword_5021348 = (__int64)"Enable memop clustering.";
  sub_C53130(&qword_5021320);
  __cxa_atexit(sub_984900, &qword_5021320, &qword_4A427C0);
  LOBYTE(v60) = 0;
  v62[0] = &v60;
  v91 = "Switch to fast cluster algorithm with the lost of some fusion opportunities";
  v92 = 75;
  LODWORD(v61) = 1;
  sub_2ECA690(&unk_5021240, "force-fast-cluster", &v61, &v91, v62);
  __cxa_atexit(sub_984900, &unk_5021240, &qword_4A427C0);
  qword_5021160 = (__int64)&unk_49DC150;
  v26 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50211B0 = 0x100000000LL;
  dword_502116C &= 0x8000u;
  qword_50211A8 = (__int64)&unk_50211B8;
  word_5021170 = 0;
  qword_5021178 = 0;
  dword_5021168 = v26;
  qword_5021180 = 0;
  qword_5021188 = 0;
  qword_5021190 = 0;
  qword_5021198 = 0;
  qword_50211A0 = 0;
  qword_50211C0 = 0;
  qword_50211C8 = (__int64)&unk_50211E0;
  qword_50211D0 = 1;
  dword_50211D8 = 0;
  byte_50211DC = 1;
  v27 = sub_C57470();
  v28 = (unsigned int)qword_50211B0;
  if ( (unsigned __int64)(unsigned int)qword_50211B0 + 1 > HIDWORD(qword_50211B0) )
  {
    v56 = v27;
    sub_C8D5F0((char *)&unk_50211B8 - 16, &unk_50211B8, (unsigned int)qword_50211B0 + 1LL, 8);
    v28 = (unsigned int)qword_50211B0;
    v27 = v56;
  }
  *(_QWORD *)(qword_50211A8 + 8 * v28) = v27;
  LODWORD(qword_50211B0) = qword_50211B0 + 1;
  qword_50211E8 = 0;
  qword_50211F0 = (__int64)&unk_49D9728;
  qword_50211F8 = 0;
  qword_5021160 = (__int64)&unk_49DBF10;
  qword_5021200 = (__int64)&unk_49DC290;
  qword_5021220 = (__int64)nullsub_24;
  qword_5021218 = (__int64)sub_984050;
  sub_C53080(&qword_5021160, "fast-cluster-threshold", 22);
  qword_5021190 = 30;
  LODWORD(qword_50211E8) = 1000;
  BYTE4(qword_50211F8) = 1;
  LODWORD(qword_50211F8) = 1000;
  LOBYTE(dword_502116C) = dword_502116C & 0x9F | 0x20;
  qword_5021188 = (__int64)"The threshold for fast cluster";
  sub_C53130(&qword_5021160);
  __cxa_atexit(sub_984970, &qword_5021160, &qword_4A427C0);
  qword_5021080 = (__int64)&unk_49DC150;
  v29 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50210D0 = 0x100000000LL;
  dword_502108C &= 0x8000u;
  qword_50210C8 = (__int64)&unk_50210D8;
  word_5021090 = 0;
  qword_5021098 = 0;
  dword_5021088 = v29;
  qword_50210A0 = 0;
  qword_50210A8 = 0;
  qword_50210B0 = 0;
  qword_50210B8 = 0;
  qword_50210C0 = 0;
  qword_50210E0 = 0;
  qword_50210E8 = (__int64)&unk_5021100;
  qword_50210F0 = 1;
  dword_50210F8 = 0;
  byte_50210FC = 1;
  v30 = sub_C57470();
  v31 = (unsigned int)qword_50210D0;
  if ( (unsigned __int64)(unsigned int)qword_50210D0 + 1 > HIDWORD(qword_50210D0) )
  {
    v57 = v30;
    sub_C8D5F0((char *)&unk_50210D8 - 16, &unk_50210D8, (unsigned int)qword_50210D0 + 1LL, 8);
    v31 = (unsigned int)qword_50210D0;
    v30 = v57;
  }
  *(_QWORD *)(qword_50210C8 + 8 * v31) = v30;
  LODWORD(qword_50210D0) = qword_50210D0 + 1;
  qword_5021108 = 0;
  qword_5021110 = (__int64)&unk_49D9728;
  qword_5021118 = 0;
  qword_5021080 = (__int64)&unk_49DBF10;
  qword_5021120 = (__int64)&unk_49DC290;
  qword_5021140 = (__int64)nullsub_24;
  qword_5021138 = (__int64)sub_984050;
  sub_C53080(&qword_5021080, "misched-resource-cutoff", 23);
  qword_50210B0 = 28;
  LODWORD(qword_5021108) = 10;
  BYTE4(qword_5021118) = 1;
  LODWORD(qword_5021118) = 10;
  LOBYTE(dword_502108C) = dword_502108C & 0x9F | 0x20;
  qword_50210A8 = (__int64)"Number of intervals to track";
  sub_C53130(&qword_5021080);
  __cxa_atexit(sub_984970, &qword_5021080, &qword_4A427C0);
  v62[0] = &v61;
  v61 = (int *)sub_2EC0BB0;
  v91 = "Machine instruction scheduler to use";
  v92 = 36;
  v60 = 1;
  sub_2ED0430(&unk_5020DA0, "misched", v62, &v60, &v91);
  __cxa_atexit(sub_2EC22D0, &unk_5020DA0, &qword_4A427C0);
  sub_2ED07E0(&unk_5020D60, "default", "Use the target's default scheduler choice.", sub_2EC0BB0);
  __cxa_atexit(sub_2EC1B50, &unk_5020D60, &qword_4A427C0);
  qword_5020C80 = (__int64)&unk_49DC150;
  v32 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_5020CFC = 1;
  word_5020C90 = 0;
  qword_5020CD0 = 0x100000000LL;
  dword_5020C8C &= 0x8000u;
  qword_5020CC8 = (__int64)&unk_5020CD8;
  qword_5020C98 = 0;
  dword_5020C88 = v32;
  qword_5020CA0 = 0;
  qword_5020CA8 = 0;
  qword_5020CB0 = 0;
  qword_5020CB8 = 0;
  qword_5020CC0 = 0;
  qword_5020CE0 = 0;
  qword_5020CE8 = (__int64)&unk_5020D00;
  qword_5020CF0 = 1;
  dword_5020CF8 = 0;
  v33 = sub_C57470();
  v34 = (unsigned int)qword_5020CD0;
  if ( (unsigned __int64)(unsigned int)qword_5020CD0 + 1 > HIDWORD(qword_5020CD0) )
  {
    v58 = v33;
    sub_C8D5F0((char *)&unk_5020CD8 - 16, &unk_5020CD8, (unsigned int)qword_5020CD0 + 1LL, 8);
    v34 = (unsigned int)qword_5020CD0;
    v33 = v58;
  }
  *(_QWORD *)(qword_5020CC8 + 8 * v34) = v33;
  LODWORD(qword_5020CD0) = qword_5020CD0 + 1;
  qword_5020D08 = 0;
  qword_5020D10 = (__int64)&unk_49D9748;
  qword_5020D18 = 0;
  qword_5020C80 = (__int64)&unk_49DC090;
  qword_5020D20 = (__int64)&unk_49DC1D0;
  qword_5020D40 = (__int64)nullsub_23;
  qword_5020D38 = (__int64)sub_984030;
  sub_C53080(&qword_5020C80, "enable-misched", 14);
  qword_5020CA8 = (__int64)"Enable the machine instruction scheduling pass.";
  LOWORD(qword_5020D18) = 257;
  LOBYTE(qword_5020D08) = 1;
  qword_5020CB0 = 47;
  LOBYTE(dword_5020C8C) = dword_5020C8C & 0x9F | 0x20;
  sub_C53130(&qword_5020C80);
  __cxa_atexit(sub_984900, &qword_5020C80, &qword_4A427C0);
  qword_5020BA0 = (__int64)&unk_49DC150;
  v35 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5020BF0 = 0x100000000LL;
  dword_5020BAC &= 0x8000u;
  word_5020BB0 = 0;
  qword_5020BE8 = (__int64)&unk_5020BF8;
  qword_5020BB8 = 0;
  dword_5020BA8 = v35;
  qword_5020BC0 = 0;
  qword_5020BC8 = 0;
  qword_5020BD0 = 0;
  qword_5020BD8 = 0;
  qword_5020BE0 = 0;
  qword_5020C00 = 0;
  qword_5020C08 = (__int64)&unk_5020C20;
  qword_5020C10 = 1;
  dword_5020C18 = 0;
  byte_5020C1C = 1;
  v36 = sub_C57470();
  v37 = (unsigned int)qword_5020BF0;
  v38 = (unsigned int)qword_5020BF0 + 1LL;
  if ( v38 > HIDWORD(qword_5020BF0) )
  {
    sub_C8D5F0((char *)&unk_5020BF8 - 16, &unk_5020BF8, v38, 8);
    v37 = (unsigned int)qword_5020BF0;
  }
  *(_QWORD *)(qword_5020BE8 + 8 * v37) = v36;
  LODWORD(qword_5020BF0) = qword_5020BF0 + 1;
  qword_5020C28 = 0;
  qword_5020C30 = (__int64)&unk_49D9748;
  qword_5020C38 = 0;
  qword_5020BA0 = (__int64)&unk_49DC090;
  qword_5020C40 = (__int64)&unk_49DC1D0;
  qword_5020C60 = (__int64)nullsub_23;
  qword_5020C58 = (__int64)sub_984030;
  sub_C53080(&qword_5020BA0, "enable-post-misched", 19);
  qword_5020BD0 = 55;
  qword_5020BC8 = (__int64)"Enable the post-ra machine instruction scheduling pass.";
  LOWORD(qword_5020C38) = 257;
  LOBYTE(qword_5020C28) = 1;
  LOBYTE(dword_5020BAC) = dword_5020BAC & 0x9F | 0x20;
  sub_C53130(&qword_5020BA0);
  __cxa_atexit(sub_984900, &qword_5020BA0, &qword_4A427C0);
  sub_2ED07E0(&unk_5020B60, "converge", "Standard converging scheduler.", sub_2ECC530);
  __cxa_atexit(sub_2EC1B50, &unk_5020B60, &qword_4A427C0);
  sub_2ED07E0(&unk_5020B20, "ilpmax", "Schedule bottom-up for max ILP", sub_2EC53C0);
  __cxa_atexit(sub_2EC1B50, &unk_5020B20, &qword_4A427C0);
  sub_2ED07E0(&unk_5020AE0, "ilpmin", "Schedule bottom-up for min ILP", sub_2EC5320);
  return __cxa_atexit(sub_2EC1B50, &unk_5020AE0, &qword_4A427C0);
}
