// Function: sub_1BA05A0
// Address: 0x1ba05a0
//
unsigned __int64 __fastcall sub_1BA05A0(__int64 a1, __int64 a2)
{
  __int64 v4; // r13
  __int64 v5; // r14
  _QWORD *v6; // rdi
  unsigned __int64 v7; // rsi
  unsigned __int64 *v8; // rax
  unsigned __int64 *v9; // r8
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // rdx
  __int64 v12; // r13
  __int64 v13; // rax
  _QWORD *v14; // rdi
  unsigned __int64 *v15; // r8
  unsigned int v16; // r15d
  _QWORD *v17; // rax
  _QWORD *v18; // rsi
  __int64 v19; // rcx
  __int64 v20; // rdx
  _QWORD *v21; // rax
  _QWORD *v22; // rdi
  __int64 *v23; // r13
  __int64 *v24; // rax
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // r15
  __int64 v28; // rdi
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  _QWORD *v37; // rdi
  unsigned __int64 result; // rax
  __int64 v39; // rsi
  __int64 *v40; // r14
  __int64 v41; // rax
  __int64 v42; // r15
  __int64 v43; // rdi
  __int64 v44; // rsi
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r9
  _QWORD *v53; // rdi
  unsigned __int64 v54; // rsi
  __int64 v55; // r12
  __int64 v56; // rbx
  unsigned __int64 *v57; // rax
  unsigned __int64 *v58; // r8
  unsigned __int64 v59; // rcx
  unsigned __int64 v60; // rdx
  __int64 *v61; // [rsp+8h] [rbp-A8h]
  unsigned __int64 *v62; // [rsp+10h] [rbp-A0h]
  __int64 v63; // [rsp+10h] [rbp-A0h]
  __int64 v64; // [rsp+10h] [rbp-A0h]
  __int64 *v65; // [rsp+18h] [rbp-98h]
  __int64 *v66; // [rsp+18h] [rbp-98h]
  __int64 v67; // [rsp+20h] [rbp-90h]
  __int64 v68; // [rsp+20h] [rbp-90h]
  __int64 v69; // [rsp+28h] [rbp-88h]
  __int64 v70; // [rsp+30h] [rbp-80h]
  __int64 v71[2]; // [rsp+40h] [rbp-70h] BYREF
  __int16 v72; // [rsp+50h] [rbp-60h]
  unsigned __int64 *v73[2]; // [rsp+60h] [rbp-50h] BYREF
  __int16 v74; // [rsp+70h] [rbp-40h]

  v4 = *(unsigned int *)(a2 + 12);
  v5 = *(unsigned int *)(a2 + 8);
  v6 = *(_QWORD **)(a2 + 184);
  v7 = *(_QWORD *)(a1 + 40);
  v71[0] = v7;
  v8 = (unsigned __int64 *)v6[9];
  v9 = v6 + 8;
  if ( !v8 )
    goto LABEL_21;
  do
  {
    while ( 1 )
    {
      v10 = v8[2];
      v11 = v8[3];
      if ( v8[4] >= v7 )
        break;
      v8 = (unsigned __int64 *)v8[3];
      if ( !v11 )
        goto LABEL_6;
    }
    v9 = v8;
    v8 = (unsigned __int64 *)v8[2];
  }
  while ( v10 );
LABEL_6:
  if ( v6 + 8 == v9 || v9[4] > v7 )
  {
LABEL_21:
    v73[0] = (unsigned __int64 *)v71;
    v9 = sub_1B99EB0(v6 + 7, v9, v73);
  }
  v12 = *(_QWORD *)(*(_QWORD *)(v9[5] + 48 * v5) + 8 * v4);
  v70 = *(_QWORD *)(v12 + 40);
  v13 = sub_157F0B0(v70);
  v14 = *(_QWORD **)(a2 + 184);
  v15 = *(unsigned __int64 **)(a1 + 40);
  v69 = v13;
  v16 = *(_DWORD *)(a2 + 8);
  v73[0] = v15;
  v17 = (_QWORD *)v14[3];
  if ( !v17 )
    goto LABEL_22;
  v18 = v14 + 2;
  do
  {
    while ( 1 )
    {
      v19 = v17[2];
      v20 = v17[3];
      if ( v17[4] >= (unsigned __int64)v15 )
        break;
      v17 = (_QWORD *)v17[3];
      if ( !v20 )
        goto LABEL_13;
    }
    v18 = v17;
    v17 = (_QWORD *)v17[2];
  }
  while ( v19 );
LABEL_13:
  if ( v14 + 2 != v18
    && v18[4] <= (unsigned __int64)v15
    && (v62 = v15,
        v21 = sub_1B975A0((__int64)(v14 + 1), (unsigned __int64 *)v73),
        v22 = v14 + 1,
        v15 = v62,
        v67 = v16,
        *(_QWORD *)(v21[5] + 8LL * v16)) )
  {
    v73[0] = v62;
    v23 = *(__int64 **)(*sub_1B99AC0(v22, (unsigned __int64 *)v73) + 8LL * v16);
    v24 = *(__int64 **)(a2 + 176);
    v72 = 257;
    v65 = v24;
    v63 = *v23;
    v74 = 257;
    v25 = sub_1648B60(64);
    v26 = v63;
    v27 = v25;
    if ( v25 )
    {
      v64 = v25;
      sub_15F1EA0(v25, v26, 53, 0, 0, 0);
      *(_DWORD *)(v27 + 56) = 2;
      sub_164B780(v27, (__int64 *)v73);
      sub_1648880(v27, *(_DWORD *)(v27 + 56), 1);
    }
    else
    {
      v64 = 0;
    }
    v28 = v65[1];
    if ( v28 )
    {
      v61 = (__int64 *)v65[2];
      sub_157E9D0(v28 + 40, v27);
      v29 = *v61;
      v30 = *(_QWORD *)(v27 + 24) & 7LL;
      *(_QWORD *)(v27 + 32) = v61;
      v29 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v27 + 24) = v29 | v30;
      *(_QWORD *)(v29 + 8) = v27 + 24;
      *v61 = *v61 & 7 | (v27 + 24);
    }
    sub_164B780(v64, v71);
    sub_12A86E0(v65, v27);
    sub_1704F80(v27, *(v23 - 9), v69, v31, v32, v33);
    sub_1704F80(v27, (__int64)v23, v70, v34, v35, v36);
    v37 = (_QWORD *)(*(_QWORD *)(a2 + 184) + 8LL);
    v73[0] = *(unsigned __int64 **)(a1 + 40);
    result = *sub_1B99AC0(v37, (unsigned __int64 *)v73);
    *(_QWORD *)(result + 8 * v67) = v27;
  }
  else
  {
LABEL_22:
    v39 = *v15;
    v72 = 257;
    v40 = *(__int64 **)(a2 + 176);
    v74 = 257;
    v41 = sub_1648B60(64);
    v42 = v41;
    if ( v41 )
    {
      v68 = v41;
      sub_15F1EA0(v41, v39, 53, 0, 0, 0);
      *(_DWORD *)(v42 + 56) = 2;
      sub_164B780(v42, (__int64 *)v73);
      sub_1648880(v42, *(_DWORD *)(v42 + 56), 1);
    }
    else
    {
      v68 = 0;
    }
    v43 = v40[1];
    if ( v43 )
    {
      v66 = (__int64 *)v40[2];
      sub_157E9D0(v43 + 40, v42);
      v44 = *v66;
      v45 = *(_QWORD *)(v42 + 24) & 7LL;
      *(_QWORD *)(v42 + 32) = v66;
      v44 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v42 + 24) = v44 | v45;
      *(_QWORD *)(v44 + 8) = v42 + 24;
      *v66 = *v66 & 7 | (v42 + 24);
    }
    sub_164B780(v68, v71);
    sub_12A86E0(v40, v42);
    v46 = sub_1599EF0(*(__int64 ***)v12);
    sub_1704F80(v42, v46, v69, v47, v48, v49);
    sub_1704F80(v42, v12, v70, v50, v51, v52);
    v53 = *(_QWORD **)(a2 + 184);
    v54 = *(_QWORD *)(a1 + 40);
    v55 = *(unsigned int *)(a2 + 12);
    v56 = *(unsigned int *)(a2 + 8);
    v71[0] = v54;
    v57 = (unsigned __int64 *)v53[9];
    v58 = v53 + 8;
    if ( !v57 )
      goto LABEL_33;
    do
    {
      while ( 1 )
      {
        v59 = v57[2];
        v60 = v57[3];
        if ( v57[4] >= v54 )
          break;
        v57 = (unsigned __int64 *)v57[3];
        if ( !v60 )
          goto LABEL_31;
      }
      v58 = v57;
      v57 = (unsigned __int64 *)v57[2];
    }
    while ( v59 );
LABEL_31:
    if ( v53 + 8 == v58 || v58[4] > v54 )
    {
LABEL_33:
      v73[0] = (unsigned __int64 *)v71;
      v58 = sub_1B99EB0(v53 + 7, v58, v73);
    }
    result = *(_QWORD *)(v58[5] + 48 * v56);
    *(_QWORD *)(result + 8 * v55) = v42;
  }
  return result;
}
