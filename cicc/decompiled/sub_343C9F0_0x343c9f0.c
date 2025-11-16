// Function: sub_343C9F0
// Address: 0x343c9f0
//
void __fastcall sub_343C9F0(
        __int64 a1,
        unsigned __int8 *a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        char a6,
        __m128i a7,
        char a8)
{
  __int64 v8; // rdx
  __int64 v9; // rax
  int v10; // edx
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r12
  int v15; // r12d
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r12
  __int64 v22; // r12
  __int64 v23; // r13
  __int64 v24; // r12
  __int64 v25; // rax
  unsigned int *v26; // rax
  __int64 v27; // rcx
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rax
  int v33; // eax
  unsigned __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // r13
  int v38; // edx
  int v39; // r12d
  _QWORD *v40; // rax
  __int64 v41; // [rsp+8h] [rbp-1428h]
  unsigned __int8 *v42; // [rsp+10h] [rbp-1420h]
  __int64 v43; // [rsp+18h] [rbp-1418h]
  __int128 v45; // [rsp+60h] [rbp-13D0h]
  __int64 v47; // [rsp+78h] [rbp-13B8h]
  unsigned __int8 *v48; // [rsp+A8h] [rbp-1388h] BYREF
  _QWORD v49[2]; // [rsp+B0h] [rbp-1380h] BYREF
  char v50; // [rsp+C0h] [rbp-1370h]
  unsigned __int64 v51[2]; // [rsp+D0h] [rbp-1360h] BYREF
  _BYTE v52[128]; // [rsp+E0h] [rbp-1350h] BYREF
  _BYTE *v53; // [rsp+160h] [rbp-12D0h]
  __int64 v54; // [rsp+168h] [rbp-12C8h]
  _BYTE v55[128]; // [rsp+170h] [rbp-12C0h] BYREF
  _BYTE *v56; // [rsp+1F0h] [rbp-1240h]
  __int64 v57; // [rsp+1F8h] [rbp-1238h]
  _BYTE v58[128]; // [rsp+200h] [rbp-1230h] BYREF
  __int64 v59; // [rsp+280h] [rbp-11B0h]
  __int64 v60; // [rsp+288h] [rbp-11A8h]
  __int64 v61; // [rsp+290h] [rbp-11A0h]
  __int64 v62; // [rsp+298h] [rbp-1198h]
  __int64 v63; // [rsp+2A0h] [rbp-1190h]
  __int64 v64; // [rsp+2A8h] [rbp-1188h]
  _QWORD v65[3]; // [rsp+2B0h] [rbp-1180h] BYREF
  unsigned __int64 v66; // [rsp+2C8h] [rbp-1168h]
  __int64 v67; // [rsp+2D0h] [rbp-1160h]
  __int64 v68; // [rsp+2D8h] [rbp-1158h]
  __int64 v69; // [rsp+2E0h] [rbp-1150h]
  unsigned __int64 v70; // [rsp+2E8h] [rbp-1148h]
  __int64 v71; // [rsp+2F0h] [rbp-1140h]
  __int64 v72; // [rsp+2F8h] [rbp-1138h]
  __int64 v73; // [rsp+300h] [rbp-1130h]
  __int64 v74; // [rsp+308h] [rbp-1128h] BYREF
  int v75; // [rsp+310h] [rbp-1120h]
  __int64 v76; // [rsp+318h] [rbp-1118h]
  _BYTE *v77; // [rsp+320h] [rbp-1110h]
  __int64 v78; // [rsp+328h] [rbp-1108h]
  _BYTE v79[1792]; // [rsp+330h] [rbp-1100h] BYREF
  _BYTE *v80; // [rsp+A30h] [rbp-A00h]
  __int64 v81; // [rsp+A38h] [rbp-9F8h]
  _BYTE v82[512]; // [rsp+A40h] [rbp-9F0h] BYREF
  _BYTE *v83; // [rsp+C40h] [rbp-7F0h]
  __int64 v84; // [rsp+C48h] [rbp-7E8h]
  _BYTE v85[1792]; // [rsp+C50h] [rbp-7E0h] BYREF
  _BYTE *v86; // [rsp+1350h] [rbp-E0h]
  __int64 v87; // [rsp+1358h] [rbp-D8h]
  _BYTE v88[64]; // [rsp+1360h] [rbp-D0h] BYREF
  __int64 v89; // [rsp+13A0h] [rbp-90h]
  __int64 v90; // [rsp+13A8h] [rbp-88h]
  int v91; // [rsp+13B0h] [rbp-80h]
  char v92; // [rsp+13D0h] [rbp-60h]
  unsigned __int8 *v93; // [rsp+13D8h] [rbp-58h]
  __int64 v94; // [rsp+13E0h] [rbp-50h]
  __int64 v95; // [rsp+13E8h] [rbp-48h]
  int v96; // [rsp+13F0h] [rbp-40h]
  __int64 v97; // [rsp+13F8h] [rbp-38h]

  v51[0] = (unsigned __int64)v52;
  v51[1] = 0x1000000000LL;
  v54 = 0x1000000000LL;
  v57 = 0x1000000000LL;
  v53 = v55;
  v66 = 0xFFFFFFFF00000020LL;
  v45 = __PAIR128__(a4, a3);
  v8 = *(_QWORD *)(a1 + 864);
  v56 = v58;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = -1;
  memset(v65, 0, sizeof(v65));
  v67 = 0;
  v77 = v79;
  v78 = 0x2000000000LL;
  v81 = 0x2000000000LL;
  v84 = 0x2000000000LL;
  v86 = v88;
  v87 = 0x400000000LL;
  v9 = *((_QWORD *)a2 + 9);
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  v72 = 0;
  v73 = v8;
  v74 = 0;
  v75 = 0;
  v76 = 0;
  v80 = v82;
  v83 = v85;
  v89 = 0;
  v90 = 0;
  v91 = 0;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v95 = -1;
  v96 = -1;
  v97 = 0;
  v49[0] = v9;
  v47 = sub_A74610(v49);
  if ( a8 )
    v43 = sub_BCB120(*(_QWORD **)(*(_QWORD *)(a1 + 864) + 64LL));
  else
    v43 = *((_QWORD *)a2 + 1);
  v10 = *a2;
  if ( v10 == 40 )
  {
    v11 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v11 = 0;
    if ( v10 != 85 )
    {
      v11 = 64;
      if ( v10 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_12;
  v12 = sub_BD2BC0((__int64)a2);
  v14 = v12 + v13;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v14 >> 4) )
LABEL_54:
      BUG();
LABEL_12:
    v18 = 0;
    goto LABEL_13;
  }
  if ( !(unsigned int)((v14 - sub_BD2BC0((__int64)a2)) >> 4) )
    goto LABEL_12;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_54;
  v15 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v16 = sub_BD2BC0((__int64)a2);
  v18 = 32LL * (unsigned int)(*(_DWORD *)(v16 + v17 - 4) - v15);
LABEL_13:
  sub_33AC4A0(
    a1,
    (__int64)v65,
    (__int64)a2,
    0,
    (32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v11 - v18) >> 5,
    v43,
    v45,
    v47,
    0);
  if ( !a6 )
    LOBYTE(v66) = (4 * (*(_DWORD *)(*((_QWORD *)a2 + 10) + 8LL) >> 8 != 0)) | v66 & 0xFB;
  if ( (a2[7] & 0x80u) != 0 )
  {
    v19 = sub_BD2BC0((__int64)a2);
    v21 = v19 + v20;
    if ( (a2[7] & 0x80u) != 0 )
      v21 -= sub_BD2BC0((__int64)a2);
    v22 = v21 >> 4;
    if ( (_DWORD)v22 )
    {
      v23 = 0;
      v24 = 16LL * (unsigned int)v22;
      while ( 1 )
      {
        v25 = 0;
        if ( (a2[7] & 0x80u) != 0 )
          v25 = sub_BD2BC0((__int64)a2);
        v26 = (unsigned int *)(v23 + v25);
        if ( !*(_DWORD *)(*(_QWORD *)v26 + 8LL) )
          break;
        v23 += 16;
        if ( v23 == v24 )
          goto LABEL_25;
      }
      v27 = 32LL * v26[2];
      v42 = &a2[v27 - 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v41 = (32LL * v26[3] - v27) >> 5;
    }
  }
LABEL_25:
  sub_3145B40((__int64)v49, *((_QWORD *)a2 + 9));
  v32 = 2882400015LL;
  if ( v50 )
    v32 = v49[1];
  v64 = v32;
  v33 = 0;
  if ( BYTE4(v49[0]) )
    v33 = v49[0];
  v96 = v33;
  v95 = 0;
  v93 = v42;
  v94 = (32 * v41) >> 5;
  v97 = a5;
  v34 = sub_343A2E0(a1, (__int64)v51, v28, v29, v30, v31, a7);
  if ( v34 )
  {
    v36 = sub_3375A10(a1, *(_QWORD *)(a1 + 864), a2, v34, v35);
    v48 = a2;
    v37 = v36;
    v39 = v38;
    v40 = sub_337DC20(a1 + 8, (__int64 *)&v48);
    *v40 = v37;
    *((_DWORD *)v40 + 2) = v39;
  }
  if ( v86 != v88 )
    _libc_free((unsigned __int64)v86);
  if ( v83 != v85 )
    _libc_free((unsigned __int64)v83);
  if ( v80 != v82 )
    _libc_free((unsigned __int64)v80);
  if ( v77 != v79 )
    _libc_free((unsigned __int64)v77);
  if ( v74 )
    sub_B91220((__int64)&v74, v74);
  if ( v70 )
    j_j___libc_free_0(v70);
  if ( v56 != v58 )
    _libc_free((unsigned __int64)v56);
  if ( v53 != v55 )
    _libc_free((unsigned __int64)v53);
  if ( (_BYTE *)v51[0] != v52 )
    _libc_free(v51[0]);
}
