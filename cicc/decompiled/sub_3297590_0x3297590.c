// Function: sub_3297590
// Address: 0x3297590
//
__int64 __fastcall sub_3297590(__int64 *a1, _QWORD *a2)
{
  _QWORD *v2; // r10
  _QWORD *v3; // rax
  __int64 v4; // r14
  __int64 v5; // r15
  __int64 v6; // r12
  __int64 v7; // r13
  unsigned int v8; // ebx
  __int64 v9; // rax
  __int64 v10; // rsi
  unsigned __int16 v11; // cx
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // r13
  char v17; // al
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r10
  bool v21; // al
  char v22; // al
  int v23; // r9d
  __int64 v24; // rax
  __int64 v25; // r12
  __int64 v26; // rax
  __int128 v27; // rax
  int v28; // r9d
  __int64 v29; // rax
  __int64 v30; // rdx
  unsigned int v31; // eax
  __int64 v32; // rdx
  unsigned int v33; // edx
  unsigned int v34; // edx
  unsigned __int64 v35; // r13
  __int64 v36; // r12
  __int64 v37; // r14
  unsigned int v38; // edx
  unsigned __int64 v39; // r13
  __int128 v40; // rax
  __int64 v41; // rax
  unsigned int v42; // edx
  int v43; // r9d
  __int128 v44; // [rsp-20h] [rbp-100h]
  __int128 v45; // [rsp-20h] [rbp-100h]
  __int128 v46; // [rsp-20h] [rbp-100h]
  __int128 v47; // [rsp-20h] [rbp-100h]
  __int128 v48; // [rsp-20h] [rbp-100h]
  __int128 v49; // [rsp-10h] [rbp-F0h]
  __int128 v50; // [rsp-10h] [rbp-F0h]
  __int128 v51; // [rsp-10h] [rbp-F0h]
  __int128 v52; // [rsp-10h] [rbp-F0h]
  __int64 v53; // [rsp+10h] [rbp-D0h]
  __int64 v54; // [rsp+18h] [rbp-C8h]
  unsigned int v55; // [rsp+18h] [rbp-C8h]
  __int64 v56; // [rsp+20h] [rbp-C0h]
  __int64 v57; // [rsp+20h] [rbp-C0h]
  unsigned __int16 v58; // [rsp+28h] [rbp-B8h]
  unsigned int v59; // [rsp+28h] [rbp-B8h]
  _QWORD *v60; // [rsp+30h] [rbp-B0h]
  __int64 v61; // [rsp+30h] [rbp-B0h]
  __int64 v62; // [rsp+30h] [rbp-B0h]
  unsigned int v63; // [rsp+30h] [rbp-B0h]
  __int64 v65; // [rsp+60h] [rbp-80h]
  unsigned int v66; // [rsp+70h] [rbp-70h] BYREF
  __int64 v67; // [rsp+78h] [rbp-68h]
  __int64 v68; // [rsp+80h] [rbp-60h] BYREF
  int v69; // [rsp+88h] [rbp-58h]
  __int64 v70; // [rsp+90h] [rbp-50h] BYREF
  __int64 v71; // [rsp+98h] [rbp-48h]
  __int64 v72; // [rsp+A0h] [rbp-40h]
  __int64 v73; // [rsp+A8h] [rbp-38h]

  v2 = a2;
  v3 = (_QWORD *)a2[5];
  v4 = *v3;
  v5 = v3[1];
  v6 = v3[5];
  v54 = *v3;
  v7 = v3[6];
  v8 = *((_DWORD *)v3 + 2);
  v9 = a2[6];
  v10 = a2[10];
  v11 = *(_WORD *)v9;
  v12 = *(_QWORD *)(v9 + 8);
  v68 = v10;
  v58 = v11;
  LOWORD(v66) = v11;
  v67 = v12;
  if ( v10 )
  {
    v60 = v2;
    sub_B96E90((__int64)&v68, v10, 1);
    v2 = v60;
  }
  v61 = (__int64)v2;
  v69 = *((_DWORD *)v2 + 18);
  v70 = v4;
  v13 = *a1;
  v71 = v5;
  v72 = v6;
  v73 = v7;
  v14 = sub_3402EA0(v13, 173, (unsigned int)&v68, v66, v67, 0, (__int64)&v70, 2);
  if ( v14 )
    goto LABEL_4;
  v17 = sub_33E2390(*a1, v4, v5, 1);
  v20 = v61;
  if ( v17 )
  {
    v22 = sub_33E2390(*a1, v6, v7, 1);
    v20 = v61;
    if ( !v22 )
    {
      *((_QWORD *)&v49 + 1) = v5;
      *(_QWORD *)&v49 = v4;
      *((_QWORD *)&v44 + 1) = v7;
      *(_QWORD *)&v44 = v6;
      v15 = sub_3411F20(*a1, 173, (unsigned int)&v68, *(_QWORD *)(v61 + 48), *(_DWORD *)(v61 + 68), v23, v44, v49);
      goto LABEL_5;
    }
  }
  if ( v58 )
  {
    if ( (unsigned __int16)(v58 - 17) > 0xD3u )
      goto LABEL_11;
  }
  else
  {
    v62 = v20;
    v21 = sub_30070B0((__int64)&v66);
    v20 = v62;
    if ( !v21 )
      goto LABEL_11;
  }
  v14 = sub_3295970(a1, v20, (__int64)&v68, v18, v19);
  if ( v14 )
  {
LABEL_4:
    v15 = v14;
    goto LABEL_5;
  }
  if ( (unsigned __int8)sub_33D1AE0(v6, 0) )
    goto LABEL_16;
LABEL_11:
  if ( (unsigned __int8)sub_33CF170(v6, v7) )
  {
    v15 = v6;
    goto LABEL_5;
  }
  if ( (unsigned __int8)sub_33CF4D0(v6, v7) )
  {
    v25 = *a1;
    v26 = sub_3263630(v54, v8);
    *(_QWORD *)&v27 = sub_3400E40(v25, v26 - 1, v66, v67, &v68);
    *((_QWORD *)&v45 + 1) = v5;
    *(_QWORD *)&v45 = v4;
    v15 = sub_3406EB0(v25, 191, (unsigned int)&v68, v66, v67, v28, v45, v27);
    goto LABEL_5;
  }
  if ( *(_DWORD *)(v54 + 24) == 51 || *(_DWORD *)(v6 + 24) == 51 )
  {
LABEL_16:
    v15 = sub_3400BD0(*a1, 0, (unsigned int)&v68, v66, v67, 0, 0);
    goto LABEL_5;
  }
  v24 = a1[1];
  if ( v58 == 1 )
  {
    if ( (*(_BYTE *)(v24 + 7087) & 0xFB) == 0 )
    {
LABEL_27:
      v15 = 0;
      goto LABEL_5;
    }
  }
  else if ( !v58
         || *(_QWORD *)(v24 + 8LL * v58 + 112) && (*(_BYTE *)(v24 + 500LL * v58 + 6587) & 0xFB) == 0
         || (unsigned __int16)(v58 - 17) <= 0xD3u )
  {
    goto LABEL_27;
  }
  v29 = sub_325F590(v58);
  v71 = v30;
  v70 = v29;
  v59 = sub_CA1930(&v70);
  v31 = sub_327FC40(*(_QWORD **)(*a1 + 64), 2 * v59);
  v56 = v32;
  v63 = v31;
  if ( !sub_328D6E0(a1[1], 0x3Au, v31) )
    goto LABEL_27;
  *((_QWORD *)&v50 + 1) = v5;
  *(_QWORD *)&v50 = v4;
  v53 = v56;
  *((_QWORD *)&v46 + 1) = v7;
  v57 = sub_33FAF80(*a1, 213, (unsigned int)&v68, v63, v56, v56, v50);
  *(_QWORD *)&v46 = v6;
  v55 = v33;
  v65 = sub_33FAF80(*a1, 213, (unsigned int)&v68, v63, v53, v53, v46);
  v35 = v34 | v7 & 0xFFFFFFFF00000000LL;
  *((_QWORD *)&v51 + 1) = v35;
  *(_QWORD *)&v51 = v65;
  *((_QWORD *)&v47 + 1) = v55 | v5 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v47 = v57;
  v36 = sub_3406EB0(*a1, 58, (unsigned int)&v68, v63, v53, v53, v47, v51);
  v37 = *a1;
  v39 = v38 | v35 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v40 = sub_3400E40(*a1, v59, v63, v53, &v68);
  *((_QWORD *)&v48 + 1) = v39;
  *(_QWORD *)&v48 = v36;
  v41 = sub_3406EB0(v37, 192, (unsigned int)&v68, v63, v53, v53, v48, v40);
  *((_QWORD *)&v52 + 1) = v42 | v39 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v52 = v41;
  v15 = sub_33FAF80(*a1, 216, (unsigned int)&v68, v66, v67, v43, v52);
LABEL_5:
  if ( v68 )
    sub_B91220((__int64)&v68, v68);
  return v15;
}
