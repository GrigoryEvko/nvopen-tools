// Function: sub_2574220
// Address: 0x2574220
//
char __fastcall sub_2574220(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // rdx
  unsigned __int64 *v8; // rbx
  unsigned __int64 *v9; // r14
  __int64 v10; // rax
  __int64 v11; // rdx
  char *v12; // r8
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  char result; // al
  unsigned __int64 *v19; // rdi
  unsigned __int64 v20; // r13
  __int64 v21; // r15
  unsigned __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdx
  unsigned __int64 *v25; // r15
  unsigned __int64 *v26; // rdi
  unsigned __int64 v27; // r13
  __int64 v28; // r15
  unsigned __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // r8
  __int64 v33; // r9
  unsigned __int64 *v34; // rdi
  unsigned __int64 v35; // r13
  __int64 v36; // r15
  unsigned __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // r8
  __int64 v40; // r9
  unsigned __int64 *v41; // rdi
  int v42; // eax
  __int64 v44; // r10
  __int64 v45; // rdx
  _QWORD *v46; // rax
  _QWORD *i; // rdx
  unsigned int v48; // ecx
  unsigned int v49; // eax
  int v50; // r12d
  unsigned int v51; // eax
  __int64 v52; // [rsp-10h] [rbp-170h]
  __int64 v53; // [rsp-10h] [rbp-170h]
  __int64 v54; // [rsp-8h] [rbp-168h]
  unsigned __int64 *v55; // [rsp+0h] [rbp-160h]
  unsigned __int64 v56; // [rsp+8h] [rbp-158h]
  char *v57; // [rsp+8h] [rbp-158h]
  char *v58; // [rsp+8h] [rbp-158h]
  char *v59; // [rsp+8h] [rbp-158h]
  char *v60; // [rsp+10h] [rbp-150h]
  __int64 v61; // [rsp+10h] [rbp-150h]
  __int64 v62; // [rsp+10h] [rbp-150h]
  __int64 v63; // [rsp+10h] [rbp-150h]
  __int64 v64; // [rsp+18h] [rbp-148h] BYREF
  char v65; // [rsp+2Fh] [rbp-131h] BYREF
  unsigned __int64 v66; // [rsp+30h] [rbp-130h] BYREF
  __int64 v67; // [rsp+38h] [rbp-128h]
  __int64 v68; // [rsp+40h] [rbp-120h] BYREF
  __int64 v69; // [rsp+48h] [rbp-118h]
  char *v70; // [rsp+50h] [rbp-110h]
  __int64 v71; // [rsp+58h] [rbp-108h]
  __int64 *v72; // [rsp+60h] [rbp-100h]
  __int64 v73; // [rsp+70h] [rbp-F0h]
  __int64 *v74; // [rsp+78h] [rbp-E8h]
  __int64 v75; // [rsp+80h] [rbp-E0h]
  __int64 v76; // [rsp+88h] [rbp-D8h]
  char *v77; // [rsp+90h] [rbp-D0h]
  __int64 v78; // [rsp+A0h] [rbp-C0h]
  __int64 *v79; // [rsp+A8h] [rbp-B8h]
  __int64 v80; // [rsp+B0h] [rbp-B0h]
  __int64 v81; // [rsp+B8h] [rbp-A8h]
  char *v82; // [rsp+C0h] [rbp-A0h]
  __int64 v83; // [rsp+D0h] [rbp-90h]
  __int64 *v84; // [rsp+D8h] [rbp-88h]
  __int64 v85; // [rsp+E0h] [rbp-80h]
  __int64 v86; // [rsp+E8h] [rbp-78h]
  char *v87; // [rsp+F0h] [rbp-70h]
  __int64 v88; // [rsp+100h] [rbp-60h]
  __int64 *v89; // [rsp+108h] [rbp-58h]
  __int64 v90; // [rsp+110h] [rbp-50h]
  __int64 v91; // [rsp+118h] [rbp-48h]
  char *v92; // [rsp+120h] [rbp-40h]

  v64 = a4;
  if ( (*(_BYTE *)(a3 + 2) & 1) != 0 )
    return 0;
  v65 = 0;
  v4 = a1;
  v5 = a2;
  if ( v64 )
    goto LABEL_3;
  v42 = *(_DWORD *)(a1 + 120);
  ++*(_QWORD *)(a1 + 104);
  v44 = a1 + 104;
  if ( !v42 )
  {
    if ( !*(_DWORD *)(a1 + 124) )
      goto LABEL_28;
    v45 = *(unsigned int *)(a1 + 128);
    if ( (unsigned int)v45 <= 0x40 )
      goto LABEL_25;
    sub_C7D6A0(*(_QWORD *)(a1 + 112), 8 * v45, 8);
    v44 = a1 + 104;
    *(_DWORD *)(a1 + 128) = 0;
LABEL_51:
    *(_QWORD *)(a1 + 112) = 0;
LABEL_27:
    *(_QWORD *)(a1 + 120) = 0;
    goto LABEL_28;
  }
  v48 = 4 * v42;
  v45 = *(unsigned int *)(a1 + 128);
  if ( (unsigned int)(4 * v42) < 0x40 )
    v48 = 64;
  if ( v48 >= (unsigned int)v45 )
  {
LABEL_25:
    v46 = *(_QWORD **)(a1 + 112);
    for ( i = &v46[v45]; i != v46; ++v46 )
      *v46 = -4096;
    goto LABEL_27;
  }
  v49 = v42 - 1;
  if ( v49 )
  {
    _BitScanReverse(&v49, v49);
    v50 = 1 << (33 - (v49 ^ 0x1F));
    if ( v50 < 64 )
      v50 = 64;
    if ( v50 == (_DWORD)v45 )
    {
      sub_2567D20(a1 + 104);
      v44 = a1 + 104;
      goto LABEL_28;
    }
  }
  else
  {
    v50 = 64;
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 112), 8 * v45, 8);
  v51 = sub_2544050(v50);
  v44 = a1 + 104;
  *(_DWORD *)(a1 + 128) = v51;
  if ( !v51 )
    goto LABEL_51;
  *(_QWORD *)(a1 + 112) = sub_C7D670(8LL * v51, 8);
  sub_2567D20(a1 + 104);
  v44 = a1 + 104;
LABEL_28:
  *(_DWORD *)(a1 + 144) = 0;
  if ( !(unsigned __int8)sub_252FFA0(a2, a3, v44, a1, &v65, 0) )
    return 0;
LABEL_3:
  v6 = *(_QWORD *)(a2 + 208);
  v7 = *(unsigned int *)(a1 + 144);
  v68 = a2;
  v8 = *(unsigned __int64 **)(a1 + 136);
  v69 = a1;
  v7 *= 8;
  v71 = v6;
  v72 = &v64;
  v9 = (unsigned __int64 *)((char *)v8 + v7);
  v10 = v7 >> 3;
  v11 = v7 >> 5;
  v70 = &v65;
  if ( v11 )
  {
    v12 = &v65;
    v55 = &v8[4 * v11];
    while ( 1 )
    {
      v60 = v12;
      v56 = *v8;
      v13 = sub_250D2C0(*v8, 0);
      v67 = v14;
      v66 = v13;
      if ( !(unsigned __int8)sub_251C230(v5, (__int64 *)&v66, v4, 0, v60, 0, 1) )
      {
        if ( *(_BYTE *)v56 != 61 )
          break;
        v19 = *(unsigned __int64 **)(v56 + 16);
        v74 = v72;
        v73 = v71;
        v75 = v68;
        v76 = v69;
        v77 = v70;
        if ( !sub_2574030(v19, 0, (__int64)v70, v15, v16, v17, v71, v72, v68, v69, v70) )
          break;
      }
      v20 = v8[1];
      v21 = v68;
      v57 = v70;
      v61 = v69;
      v22 = sub_250D2C0(v20, 0);
      v67 = v23;
      v66 = v22;
      if ( !(unsigned __int8)sub_251C230(v21, (__int64 *)&v66, v61, 0, v57, 0, 1) )
      {
        v25 = v8 + 1;
        if ( *(_BYTE *)v20 != 61 )
          return v9 == v25;
        v26 = *(unsigned __int64 **)(v20 + 16);
        v79 = v72;
        v78 = v71;
        v80 = v68;
        v81 = v69;
        v82 = v70;
        if ( !sub_2574030(v26, 0, v24, (__int64)v70, v52, v54, v71, v72, v68, v69, v70) )
          return v9 == v25;
      }
      v27 = v8[2];
      v28 = v68;
      v58 = v70;
      v62 = v69;
      v29 = sub_250D2C0(v27, 0);
      v67 = v30;
      v66 = v29;
      if ( !(unsigned __int8)sub_251C230(v28, (__int64 *)&v66, v62, 0, v58, 0, 1) )
      {
        v25 = v8 + 2;
        if ( *(_BYTE *)v27 != 61 )
          return v9 == v25;
        v34 = *(unsigned __int64 **)(v27 + 16);
        v84 = v72;
        v83 = v71;
        v85 = v68;
        v86 = v69;
        v87 = v70;
        if ( !sub_2574030(v34, 0, v31, (__int64)v70, v32, v33, v71, v72, v68, v69, v70) )
          return v9 == v25;
      }
      v35 = v8[3];
      v36 = v68;
      v59 = v70;
      v63 = v69;
      v37 = sub_250D2C0(v35, 0);
      v67 = v38;
      v66 = v37;
      if ( !(unsigned __int8)sub_251C230(v36, (__int64 *)&v66, v63, 0, v59, 0, 1) )
      {
        v25 = v8 + 3;
        if ( *(_BYTE *)v35 != 61 )
          return v9 == v25;
        v41 = *(unsigned __int64 **)(v35 + 16);
        v89 = v72;
        v88 = v71;
        v90 = v68;
        v91 = v69;
        v92 = v70;
        if ( !sub_2574030(v41, 0, v53, (__int64)v70, v39, v40, v71, v72, v68, v69, v70) )
          return v9 == v25;
      }
      v8 += 4;
      if ( v55 == v8 )
      {
        v10 = v9 - v8;
        goto LABEL_31;
      }
      v5 = v68;
      v12 = v70;
      v4 = v69;
    }
    return v9 == v8;
  }
LABEL_31:
  if ( v10 == 2 )
    goto LABEL_46;
  if ( v10 == 3 )
  {
    if ( !sub_2574150(&v68, *v8) )
      return v9 == v8;
    ++v8;
LABEL_46:
    if ( sub_2574150(&v68, *v8) )
    {
      ++v8;
      goto LABEL_48;
    }
    return v9 == v8;
  }
  if ( v10 != 1 )
    return 1;
LABEL_48:
  result = sub_2574150(&v68, *v8);
  if ( !result )
    return v9 == v8;
  return result;
}
