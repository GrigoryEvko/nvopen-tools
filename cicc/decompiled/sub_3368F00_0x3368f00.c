// Function: sub_3368F00
// Address: 0x3368f00
//
__int64 __fastcall sub_3368F00(__int128 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r15
  __int64 v10; // r14
  int v11; // r9d
  __int128 v12; // rax
  int v13; // r9d
  __int128 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // rax
  int v17; // eax
  int v18; // edx
  __int128 v19; // rax
  int v20; // r9d
  __int64 v21; // r14
  unsigned int v22; // edx
  unsigned __int64 v23; // r15
  __int128 v24; // rax
  int v25; // r9d
  __int128 v26; // rax
  __int128 v27; // rax
  __int128 v28; // rax
  int v29; // r9d
  __int128 v30; // rax
  __int128 v31; // rax
  __int128 v32; // rax
  int v33; // r9d
  __int128 v34; // rax
  __int128 v35; // rax
  __int128 v36; // rax
  int v37; // r9d
  __int128 v38; // rax
  __int128 v39; // rax
  __int128 v40; // rax
  int v41; // r9d
  __int128 v42; // rax
  __int128 v43; // rax
  __int128 v44; // rax
  int v45; // r9d
  __int128 v46; // rax
  __int128 v47; // rax
  int v48; // r9d
  __int128 v49; // rax
  int v50; // r9d
  int v51; // r9d
  __int128 v53; // rax
  int v54; // r9d
  __int128 v55; // rax
  __int128 v56; // rax
  __int128 v57; // rax
  int v58; // r9d
  __int128 v59; // rax
  __int128 v60; // rax
  __int128 v61; // rax
  int v62; // r9d
  __int128 v63; // rax
  __int128 v64; // rax
  int v65; // r9d
  __int128 v66; // rax
  __int128 v67; // rax
  __int128 v68; // rax
  int v69; // r9d
  __int128 v70; // rax
  __int128 v71; // rax
  __int128 v72; // [rsp-20h] [rbp-D0h]
  __int128 v73; // [rsp-10h] [rbp-C0h]
  __int128 v74; // [rsp+20h] [rbp-90h]
  __int128 v75; // [rsp+20h] [rbp-90h]
  __int128 v76; // [rsp+20h] [rbp-90h]
  __int128 v77; // [rsp+20h] [rbp-90h]
  __int128 v78; // [rsp+20h] [rbp-90h]
  __int128 v79; // [rsp+20h] [rbp-90h]
  __int128 v80; // [rsp+20h] [rbp-90h]
  __int128 v81; // [rsp+20h] [rbp-90h]
  __int128 v82; // [rsp+30h] [rbp-80h]
  __int128 v83; // [rsp+30h] [rbp-80h]
  __int128 v84; // [rsp+30h] [rbp-80h]

  v7 = sub_33FAF80(a3, 226, a2, 7, 0, a5);
  v9 = v8;
  v10 = v7;
  *(_QWORD *)&v12 = sub_33FAF80(a3, 220, a2, 12, 0, v11);
  *(_QWORD *)&v14 = sub_3406EB0(a3, 97, a2, 12, 0, v13, a1, v12);
  v15 = *(_QWORD *)(a3 + 16);
  v82 = v14;
  v16 = sub_2E79000(*(__int64 **)(a3 + 40));
  v17 = sub_2FE6750(v15, 7, 0, v16);
  *(_QWORD *)&v19 = sub_3400BD0(a3, 23, a2, v17, v18, 0, 0);
  *((_QWORD *)&v72 + 1) = v9;
  *(_QWORD *)&v72 = v10;
  v21 = sub_3406EB0(a3, 190, a2, 7, 0, v20, v72, v19);
  v23 = v22 | v9 & 0xFFFFFFFF00000000LL;
  if ( (unsigned int)dword_5039408 <= 6 )
  {
    *(_QWORD *)&v64 = sub_3368DE0(a3, 0x3E814304u, a2);
    *(_QWORD *)&v66 = sub_3406EB0(a3, 98, a2, 12, 0, v65, v82, v64);
    v81 = v66;
    *(_QWORD *)&v67 = sub_3368DE0(a3, 0x3F3C50C8u, a2);
    *(_QWORD *)&v68 = sub_3406EB0(a3, 96, a2, 12, 0, DWORD2(v81), v81, v67);
    *(_QWORD *)&v70 = sub_3406EB0(a3, 98, a2, 12, 0, v69, v68, v82);
    v84 = v70;
    *(_QWORD *)&v71 = sub_3368DE0(a3, 0x3F7F5E7Eu, a2);
    sub_3406EB0(a3, 96, a2, 12, 0, DWORD2(v84), v84, v71);
  }
  else
  {
    if ( (unsigned int)dword_5039408 <= 0xC )
    {
      *(_QWORD *)&v53 = sub_3368DE0(a3, 0x3DA235E3u, a2);
      *(_QWORD *)&v55 = sub_3406EB0(a3, 98, a2, 12, 0, v54, v82, v53);
      v79 = v55;
      *(_QWORD *)&v56 = sub_3368DE0(a3, 0x3E65B8F3u, a2);
      *(_QWORD *)&v57 = sub_3406EB0(a3, 96, a2, 12, 0, DWORD2(v79), v79, v56);
      *(_QWORD *)&v59 = sub_3406EB0(a3, 98, a2, 12, 0, v58, v57, v82);
      v80 = v59;
      *(_QWORD *)&v60 = sub_3368DE0(a3, 0x3F324B07u, a2);
      *(_QWORD *)&v61 = sub_3406EB0(a3, 96, a2, 12, 0, DWORD2(v80), v80, v60);
      *(_QWORD *)&v63 = sub_3406EB0(a3, 98, a2, 12, 0, v62, v61, v82);
      v83 = v63;
      *(_QWORD *)&v47 = sub_3368DE0(a3, 0x3F7FF8FDu, a2);
    }
    else
    {
      *(_QWORD *)&v24 = sub_3368DE0(a3, 0x3924B03Eu, a2);
      *(_QWORD *)&v26 = sub_3406EB0(a3, 98, a2, 12, 0, v25, v82, v24);
      v74 = v26;
      *(_QWORD *)&v27 = sub_3368DE0(a3, 0x3AB24B87u, a2);
      *(_QWORD *)&v28 = sub_3406EB0(a3, 96, a2, 12, 0, DWORD2(v74), v74, v27);
      *(_QWORD *)&v30 = sub_3406EB0(a3, 98, a2, 12, 0, v29, v28, v82);
      v75 = v30;
      *(_QWORD *)&v31 = sub_3368DE0(a3, 0x3C1D8C17u, a2);
      *(_QWORD *)&v32 = sub_3406EB0(a3, 96, a2, 12, 0, DWORD2(v75), v75, v31);
      *(_QWORD *)&v34 = sub_3406EB0(a3, 98, a2, 12, 0, v33, v32, v82);
      v76 = v34;
      *(_QWORD *)&v35 = sub_3368DE0(a3, 0x3D634A1Du, a2);
      *(_QWORD *)&v36 = sub_3406EB0(a3, 96, a2, 12, 0, DWORD2(v76), v76, v35);
      *(_QWORD *)&v38 = sub_3406EB0(a3, 98, a2, 12, 0, v37, v36, v82);
      v77 = v38;
      *(_QWORD *)&v39 = sub_3368DE0(a3, 0x3E75FE14u, a2);
      *(_QWORD *)&v40 = sub_3406EB0(a3, 96, a2, 12, 0, DWORD2(v77), v77, v39);
      *(_QWORD *)&v42 = sub_3406EB0(a3, 98, a2, 12, 0, v41, v40, v82);
      v78 = v42;
      *(_QWORD *)&v43 = sub_3368DE0(a3, 0x3F317234u, a2);
      *(_QWORD *)&v44 = sub_3406EB0(a3, 96, a2, 12, 0, DWORD2(v78), v78, v43);
      *(_QWORD *)&v46 = sub_3406EB0(a3, 98, a2, 12, 0, v45, v44, v82);
      v83 = v46;
      *(_QWORD *)&v47 = sub_3368DE0(a3, 0x3F800000u, a2);
    }
    sub_3406EB0(a3, 96, a2, 12, 0, DWORD2(v83), v83, v47);
  }
  *(_QWORD *)&v49 = sub_33FAF80(a3, 234, a2, 7, 0, v48);
  *((_QWORD *)&v73 + 1) = v23;
  *(_QWORD *)&v73 = v21;
  sub_3406EB0(a3, 56, a2, 7, 0, v50, v49, v73);
  return sub_33FAF80(a3, 234, a2, 12, 0, v51);
}
