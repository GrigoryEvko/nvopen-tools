// Function: ctor_426_0
// Address: 0x535270
//
int ctor_426_0()
{
  int v0; // edx
  int v1; // ecx
  int v2; // r8d
  int v3; // r9d
  int v4; // r8d
  int v5; // r9d
  __int128 v7; // [rsp-80h] [rbp-2B0h]
  __int128 v8; // [rsp-80h] [rbp-2B0h]
  __int128 v9; // [rsp-70h] [rbp-2A0h]
  __int128 v10; // [rsp-70h] [rbp-2A0h]
  __int128 v11; // [rsp-58h] [rbp-288h]
  __int128 v12; // [rsp-58h] [rbp-288h]
  __int128 v13; // [rsp-48h] [rbp-278h]
  __int128 v14; // [rsp-48h] [rbp-278h]
  __int128 v15; // [rsp-30h] [rbp-260h]
  __int128 v16; // [rsp-30h] [rbp-260h]
  __int128 v17; // [rsp-20h] [rbp-250h]
  __int128 v18; // [rsp-20h] [rbp-250h]
  int v19; // [rsp+10h] [rbp-220h] BYREF
  int v20; // [rsp+14h] [rbp-21Ch] BYREF
  int *v21; // [rsp+18h] [rbp-218h] BYREF
  _QWORD v22[4]; // [rsp+20h] [rbp-210h] BYREF
  __int64 v23; // [rsp+40h] [rbp-1F0h]
  const char *v24; // [rsp+48h] [rbp-1E8h]
  __int64 v25; // [rsp+50h] [rbp-1E0h]
  _QWORD v26[2]; // [rsp+60h] [rbp-1D0h] BYREF
  __int64 v27; // [rsp+70h] [rbp-1C0h]
  const char *v28; // [rsp+78h] [rbp-1B8h]
  __int64 v29; // [rsp+80h] [rbp-1B0h]
  char *v30; // [rsp+90h] [rbp-1A0h] BYREF
  __int64 v31; // [rsp+98h] [rbp-198h]
  __int64 v32; // [rsp+A0h] [rbp-190h]
  const char *v33; // [rsp+A8h] [rbp-188h]
  __int64 v34; // [rsp+B0h] [rbp-180h]
  __int64 v35; // [rsp+C8h] [rbp-168h]
  __int64 v36; // [rsp+D0h] [rbp-160h]
  const char *v37; // [rsp+D8h] [rbp-158h]
  __int64 v38; // [rsp+E0h] [rbp-150h]
  __int64 v39; // [rsp+F8h] [rbp-138h]
  __int64 v40; // [rsp+100h] [rbp-130h]
  const char *v41; // [rsp+108h] [rbp-128h]
  __int64 v42; // [rsp+110h] [rbp-120h]
  __int64 v43; // [rsp+128h] [rbp-108h]
  __int64 v44; // [rsp+130h] [rbp-100h]
  const char *v45; // [rsp+138h] [rbp-F8h]
  __int64 v46; // [rsp+140h] [rbp-F0h]
  _QWORD v47[2]; // [rsp+150h] [rbp-E0h] BYREF
  _BYTE v48[208]; // [rsp+160h] [rbp-D0h] BYREF

  sub_D95050(&qword_4FF3120, 0, 0);
  qword_4FF31A8 = 0;
  qword_4FF31B8 = 0;
  qword_4FF31B0 = (__int64)&unk_49D9748;
  qword_4FF3120 = (__int64)&unk_49DC090;
  qword_4FF31C0 = (__int64)&unk_49DC1D0;
  qword_4FF31E0 = (__int64)nullsub_23;
  qword_4FF31D8 = (__int64)sub_984030;
  sub_C53080(&qword_4FF3120, "lowertypetests-avoid-reuse", 26);
  qword_4FF3150 = 56;
  qword_4FF3148 = (__int64)"Try to avoid reuse of byte array addresses using aliases";
  LOBYTE(qword_4FF31A8) = 1;
  byte_4FF312C = byte_4FF312C & 0x9F | 0x20;
  LOWORD(qword_4FF31B8) = 257;
  sub_C53130(&qword_4FF3120);
  __cxa_atexit(sub_984900, &qword_4FF3120, &qword_4A427C0);
  v45 = "Export typeid resolutions to summary and globals";
  v41 = "Import typeid resolutions from summary and globals";
  v37 = "Do nothing";
  v43 = 6;
  LODWORD(v44) = 2;
  v46 = 48;
  *((_QWORD *)&v17 + 1) = "Export typeid resolutions to summary and globals";
  *(_QWORD *)&v17 = v44;
  *((_QWORD *)&v15 + 1) = 6;
  *(_QWORD *)&v15 = "export";
  v39 = 6;
  LODWORD(v40) = 1;
  v42 = 50;
  LODWORD(v26[0]) = 1;
  v35 = 4;
  *((_QWORD *)&v13 + 1) = "Import typeid resolutions from summary and globals";
  LODWORD(v36) = 0;
  *(_QWORD *)&v13 = v40;
  v38 = 10;
  *((_QWORD *)&v11 + 1) = 6;
  *(_QWORD *)&v11 = "import";
  *((_QWORD *)&v9 + 1) = "Do nothing";
  *(_QWORD *)&v9 = v36;
  *((_QWORD *)&v7 + 1) = 4;
  *(_QWORD *)&v7 = "none";
  sub_22735E0((unsigned int)v47, (unsigned int)&qword_4FF3120, v0, v1, v2, v3, v7, v9, 10, v11, v13, 50, v15, v17, 48);
  v30 = "What to do with the summary when running this pass";
  v31 = 50;
  sub_263C1C0(&unk_4FF2EC0, "lowertypetests-summary-action", &v30, v47, v26);
  if ( (_BYTE *)v47[0] != v48 )
    _libc_free(v47[0], "lowertypetests-summary-action");
  __cxa_atexit(sub_261AD80, &unk_4FF2EC0, &qword_4A427C0);
  sub_D95050(&qword_4FF2DC0, 0, 0);
  qword_4FF2E48 = (__int64)&byte_4FF2E58;
  qword_4FF2E70 = (__int64)&byte_4FF2E80;
  qword_4FF2E50 = 0;
  byte_4FF2E58 = 0;
  qword_4FF2E68 = (__int64)&unk_49DC130;
  qword_4FF2DC0 = (__int64)&unk_49DC010;
  qword_4FF2E78 = 0;
  qword_4FF2EB8 = (__int64)nullsub_92;
  byte_4FF2E80 = 0;
  qword_4FF2EB0 = (__int64)sub_BC4D70;
  byte_4FF2E90 = 0;
  qword_4FF2E98 = (__int64)&unk_49DC350;
  sub_C53080(&qword_4FF2DC0, "lowertypetests-read-summary", 27);
  qword_4FF2DF0 = 53;
  qword_4FF2DE8 = (__int64)"Read summary from given YAML file before running pass";
  byte_4FF2DCC = byte_4FF2DCC & 0x9F | 0x20;
  sub_C53130(&qword_4FF2DC0);
  __cxa_atexit(sub_BC5A40, &qword_4FF2DC0, &qword_4A427C0);
  sub_D95050(&qword_4FF2CC0, 0, 0);
  qword_4FF2D48 = (__int64)&byte_4FF2D58;
  qword_4FF2D70 = (__int64)&byte_4FF2D80;
  qword_4FF2D68 = (__int64)&unk_49DC130;
  qword_4FF2DB8 = (__int64)nullsub_92;
  qword_4FF2CC0 = (__int64)&unk_49DC010;
  qword_4FF2DB0 = (__int64)sub_BC4D70;
  qword_4FF2D50 = 0;
  byte_4FF2D58 = 0;
  qword_4FF2D78 = 0;
  byte_4FF2D80 = 0;
  byte_4FF2D90 = 0;
  qword_4FF2D98 = (__int64)&unk_49DC350;
  sub_C53080(&qword_4FF2CC0, "lowertypetests-write-summary", 28);
  qword_4FF2CF0 = 51;
  qword_4FF2CE8 = (__int64)"Write summary to given YAML file after running pass";
  byte_4FF2CCC = byte_4FF2CCC & 0x9F | 0x20;
  sub_C53130(&qword_4FF2CC0);
  __cxa_atexit(sub_BC5A40, &qword_4FF2CC0, &qword_4A427C0);
  v21 = &v20;
  v33 = "Drop all type test sequences";
  v28 = "Drop type test assume sequences";
  v30 = "all";
  v26[0] = "assume";
  v24 = "Do not drop any type tests";
  v31 = 3;
  LODWORD(v32) = 2;
  v34 = 28;
  LODWORD(v27) = 1;
  v29 = 31;
  *((_QWORD *)&v18 + 1) = "Drop all type test sequences";
  v20 = 0;
  *(_QWORD *)&v18 = v32;
  v19 = 1;
  *((_QWORD *)&v16 + 1) = 3;
  v26[1] = 6;
  *(_QWORD *)&v16 = "all";
  v22[3] = 4;
  LODWORD(v23) = 0;
  *((_QWORD *)&v14 + 1) = "Drop type test assume sequences";
  v25 = 26;
  *(_QWORD *)&v14 = v27;
  *((_QWORD *)&v12 + 1) = 6;
  *(_QWORD *)&v12 = "assume";
  *((_QWORD *)&v10 + 1) = "Do not drop any type tests";
  *(_QWORD *)&v10 = v23;
  *((_QWORD *)&v8 + 1) = 4;
  *(_QWORD *)&v8 = "none";
  sub_22735E0(
    (unsigned int)v47,
    (unsigned int)&qword_4FF2CC0,
    (unsigned int)"all",
    (unsigned int)"Do not drop any type tests",
    v4,
    v5,
    v8,
    v10,
    26,
    v12,
    v14,
    31,
    v16,
    v18,
    28);
  v22[0] = "Simply drop type test sequences";
  v22[1] = 31;
  sub_263C640(&unk_4FF2A60, "lowertypetests-drop-type-tests", v22, v47, &v19, &v21);
  if ( (_BYTE *)v47[0] != v48 )
    _libc_free(v47[0], "lowertypetests-drop-type-tests");
  return __cxa_atexit(sub_261ACF0, &unk_4FF2A60, &qword_4A427C0);
}
