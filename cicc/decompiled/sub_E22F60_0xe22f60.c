// Function: sub_E22F60
// Address: 0xe22f60
//
unsigned __int64 __fastcall sub_E22F60(__int64 a1, size_t *a2)
{
  char *v4; // rdx
  size_t v5; // rsi
  char v6; // al
  __int64 v7; // rdi
  _QWORD *v8; // rdx
  unsigned __int64 result; // rax
  unsigned __int64 *v10; // rax
  unsigned __int64 *v11; // r12
  unsigned __int64 v12; // rdx
  _QWORD *v13; // rdx
  _QWORD *v14; // rdx
  _QWORD *v15; // rdx
  _QWORD *v16; // rdx
  _QWORD *v17; // rdx
  _QWORD *v18; // rdx
  _QWORD *v19; // rdx
  _QWORD *v20; // rdx
  char v21; // al
  _QWORD *v22; // rdx
  _QWORD *v23; // rdx
  _QWORD *v24; // rdx
  _QWORD *v25; // rdx
  _QWORD *v26; // rdx
  _QWORD *v27; // rdx
  _QWORD *v28; // rdx
  _QWORD *v29; // rdx
  _QWORD *v30; // rdx
  _QWORD *v31; // rdx
  _QWORD *v32; // rdx
  _QWORD *v33; // rdx
  _QWORD *v34; // rdx
  _QWORD *v35; // rdx
  unsigned __int64 *v36; // rax
  unsigned __int64 *v37; // r12
  unsigned __int64 v38; // rdx
  unsigned __int64 *v39; // rax
  unsigned __int64 *v40; // r12
  unsigned __int64 v41; // rdx
  unsigned __int64 *v42; // rax
  unsigned __int64 *v43; // r12
  unsigned __int64 v44; // rdx
  unsigned __int64 *v45; // rax
  unsigned __int64 *v46; // r12
  unsigned __int64 v47; // rdx
  unsigned __int64 *v48; // rax
  unsigned __int64 *v49; // r12
  unsigned __int64 v50; // rdx
  unsigned __int64 *v51; // rax
  unsigned __int64 *v52; // r12
  unsigned __int64 v53; // rdx
  unsigned __int64 *v54; // rax
  unsigned __int64 *v55; // r12
  unsigned __int64 v56; // rdx
  unsigned __int64 *v57; // rax
  unsigned __int64 *v58; // r12
  unsigned __int64 v59; // rdx
  unsigned __int64 *v60; // rax
  unsigned __int64 *v61; // r12
  unsigned __int64 v62; // rdx
  unsigned __int64 *v63; // rax
  unsigned __int64 *v64; // r12
  unsigned __int64 v65; // rdx
  unsigned __int64 *v66; // rax
  unsigned __int64 *v67; // r12
  unsigned __int64 v68; // rdx
  unsigned __int64 *v69; // rax
  unsigned __int64 *v70; // r12
  unsigned __int64 v71; // rdx
  unsigned __int64 *v72; // rax
  unsigned __int64 *v73; // r12
  unsigned __int64 v74; // rdx
  unsigned __int64 *v75; // rax
  unsigned __int64 *v76; // r12
  unsigned __int64 v77; // rdx
  unsigned __int64 *v78; // rax
  unsigned __int64 *v79; // r12
  unsigned __int64 v80; // rdx
  unsigned __int64 *v81; // rax
  unsigned __int64 *v82; // r12
  unsigned __int64 v83; // rdx
  unsigned __int64 *v84; // rax
  unsigned __int64 *v85; // r12
  unsigned __int64 v86; // rdx
  unsigned __int64 *v87; // rax
  unsigned __int64 *v88; // r12
  unsigned __int64 v89; // rdx
  unsigned __int64 *v90; // rax
  unsigned __int64 *v91; // r12
  unsigned __int64 v92; // rdx
  unsigned __int64 *v93; // rax
  unsigned __int64 *v94; // r12
  unsigned __int64 v95; // rdx
  unsigned __int64 *v96; // rax
  unsigned __int64 *v97; // r12
  unsigned __int64 v98; // rdx
  unsigned __int64 *v99; // rax
  unsigned __int64 *v100; // r12
  unsigned __int64 v101; // rdx

  if ( (unsigned __int8)sub_E20730(a2, 3u, &unk_3F7C540) )
  {
    v8 = *(_QWORD **)(a1 + 16);
    result = (*v8 + v8[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
    v8[1] = result - *v8 + 24;
    if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
    {
      v10 = (unsigned __int64 *)sub_22077B0(32);
      v11 = v10;
      if ( v10 )
      {
        *v10 = 0;
        v10[1] = 0;
        v10[2] = 0;
        v10[3] = 0;
      }
      result = sub_2207820(4096);
      v12 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v11;
      *v11 = result;
      v11[3] = v12;
      v11[2] = 4096;
      v11[1] = 24;
      if ( !result )
        return result;
    }
    else if ( !result )
    {
      return 0;
    }
    *(_DWORD *)(result + 8) = 2;
    *(_BYTE *)(result + 12) = 0;
    *(_DWORD *)(result + 16) = 20;
    *(_QWORD *)result = &unk_49E0E88;
    return result;
  }
  v4 = (char *)a2[1];
  v5 = *a2;
  v6 = *v4;
  v7 = *a2 - 1;
  a2[1] = (size_t)(v4 + 1);
  *a2 = v7;
  switch ( v6 )
  {
    case 'C':
      v26 = *(_QWORD **)(a1 + 16);
      result = (*v26 + v26[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v26[1] = result - *v26 + 24;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        if ( !result )
          return 0;
        goto LABEL_52;
      }
      v72 = (unsigned __int64 *)sub_22077B0(32);
      v73 = v72;
      if ( v72 )
      {
        *v72 = 0;
        v72[1] = 0;
        v72[2] = 0;
        v72[3] = 0;
      }
      result = sub_2207820(4096);
      v74 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v73;
      *v73 = result;
      v73[3] = v74;
      v73[2] = 4096;
      v73[1] = 24;
      if ( result )
      {
LABEL_52:
        *(_DWORD *)(result + 8) = 2;
        *(_BYTE *)(result + 12) = 0;
        *(_DWORD *)(result + 16) = 3;
        *(_QWORD *)result = &unk_49E0E88;
        return result;
      }
      return result;
    case 'D':
      v13 = *(_QWORD **)(a1 + 16);
      result = (*v13 + v13[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v13[1] = result - *v13 + 24;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        if ( !result )
          return 0;
        goto LABEL_14;
      }
      v36 = (unsigned __int64 *)sub_22077B0(32);
      v37 = v36;
      if ( v36 )
      {
        *v36 = 0;
        v36[1] = 0;
        v36[2] = 0;
        v36[3] = 0;
      }
      result = sub_2207820(4096);
      v38 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v37;
      *v37 = result;
      v37[3] = v38;
      v37[2] = 4096;
      v37[1] = 24;
      if ( result )
      {
LABEL_14:
        *(_DWORD *)(result + 8) = 2;
        *(_BYTE *)(result + 12) = 0;
        *(_DWORD *)(result + 16) = 2;
        *(_QWORD *)result = &unk_49E0E88;
        return result;
      }
      return result;
    case 'E':
      v25 = *(_QWORD **)(a1 + 16);
      result = (*v25 + v25[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v25[1] = result - *v25 + 24;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        if ( !result )
          return 0;
        goto LABEL_49;
      }
      v69 = (unsigned __int64 *)sub_22077B0(32);
      v70 = v69;
      if ( v69 )
      {
        *v69 = 0;
        v69[1] = 0;
        v69[2] = 0;
        v69[3] = 0;
      }
      result = sub_2207820(4096);
      v71 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v70;
      *v70 = result;
      v70[3] = v71;
      v70[2] = 4096;
      v70[1] = 24;
      if ( result )
      {
LABEL_49:
        *(_DWORD *)(result + 8) = 2;
        *(_BYTE *)(result + 12) = 0;
        *(_DWORD *)(result + 16) = 4;
        *(_QWORD *)result = &unk_49E0E88;
        return result;
      }
      return result;
    case 'F':
      v20 = *(_QWORD **)(a1 + 16);
      result = (*v20 + v20[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v20[1] = result - *v20 + 24;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        if ( !result )
          return 0;
        goto LABEL_35;
      }
      v57 = (unsigned __int64 *)sub_22077B0(32);
      v58 = v57;
      if ( v57 )
      {
        *v57 = 0;
        v57[1] = 0;
        v57[2] = 0;
        v57[3] = 0;
      }
      result = sub_2207820(4096);
      v59 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v58;
      *v58 = result;
      v58[3] = v59;
      v58[2] = 4096;
      v58[1] = 24;
      if ( result )
      {
LABEL_35:
        *(_DWORD *)(result + 8) = 2;
        *(_BYTE *)(result + 12) = 0;
        *(_DWORD *)(result + 16) = 8;
        *(_QWORD *)result = &unk_49E0E88;
        return result;
      }
      return result;
    case 'G':
      v24 = *(_QWORD **)(a1 + 16);
      result = (*v24 + v24[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v24[1] = result - *v24 + 24;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        if ( !result )
          return 0;
        goto LABEL_46;
      }
      v60 = (unsigned __int64 *)sub_22077B0(32);
      v61 = v60;
      if ( v60 )
      {
        *v60 = 0;
        v60[1] = 0;
        v60[2] = 0;
        v60[3] = 0;
      }
      result = sub_2207820(4096);
      v62 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v61;
      *v61 = result;
      v61[3] = v62;
      v61[2] = 4096;
      v61[1] = 24;
      if ( result )
      {
LABEL_46:
        *(_DWORD *)(result + 8) = 2;
        *(_BYTE *)(result + 12) = 0;
        *(_DWORD *)(result + 16) = 9;
        *(_QWORD *)result = &unk_49E0E88;
        return result;
      }
      return result;
    case 'H':
      v22 = *(_QWORD **)(a1 + 16);
      result = (*v22 + v22[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v22[1] = result - *v22 + 24;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        if ( !result )
          return 0;
        goto LABEL_40;
      }
      v54 = (unsigned __int64 *)sub_22077B0(32);
      v55 = v54;
      if ( v54 )
      {
        *v54 = 0;
        v54[1] = 0;
        v54[2] = 0;
        v54[3] = 0;
      }
      result = sub_2207820(4096);
      v56 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v55;
      *v55 = result;
      v55[3] = v56;
      v55[2] = 4096;
      v55[1] = 24;
      if ( result )
      {
LABEL_40:
        *(_DWORD *)(result + 8) = 2;
        *(_BYTE *)(result + 12) = 0;
        *(_DWORD *)(result + 16) = 10;
        *(_QWORD *)result = &unk_49E0E88;
        return result;
      }
      return result;
    case 'I':
      v14 = *(_QWORD **)(a1 + 16);
      result = (*v14 + v14[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v14[1] = result - *v14 + 24;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        if ( !result )
          return 0;
        goto LABEL_17;
      }
      v39 = (unsigned __int64 *)sub_22077B0(32);
      v40 = v39;
      if ( v39 )
      {
        *v39 = 0;
        v39[1] = 0;
        v39[2] = 0;
        v39[3] = 0;
      }
      result = sub_2207820(4096);
      v41 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v40;
      *v40 = result;
      v40[3] = v41;
      v40[2] = 4096;
      v40[1] = 24;
      if ( result )
      {
LABEL_17:
        *(_DWORD *)(result + 8) = 2;
        *(_BYTE *)(result + 12) = 0;
        *(_DWORD *)(result + 16) = 11;
        *(_QWORD *)result = &unk_49E0E88;
        return result;
      }
      return result;
    case 'J':
      v15 = *(_QWORD **)(a1 + 16);
      result = (*v15 + v15[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v15[1] = result - *v15 + 24;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        if ( !result )
          return 0;
        goto LABEL_20;
      }
      v45 = (unsigned __int64 *)sub_22077B0(32);
      v46 = v45;
      if ( v45 )
      {
        *v45 = 0;
        v45[1] = 0;
        v45[2] = 0;
        v45[3] = 0;
      }
      result = sub_2207820(4096);
      v47 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v46;
      *v46 = result;
      v46[3] = v47;
      v46[2] = 4096;
      v46[1] = 24;
      if ( result )
      {
LABEL_20:
        *(_DWORD *)(result + 8) = 2;
        *(_BYTE *)(result + 12) = 0;
        *(_DWORD *)(result + 16) = 12;
        *(_QWORD *)result = &unk_49E0E88;
        return result;
      }
      return result;
    case 'K':
      v17 = *(_QWORD **)(a1 + 16);
      result = (*v17 + v17[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v17[1] = result - *v17 + 24;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        if ( !result )
          return 0;
        goto LABEL_26;
      }
      v48 = (unsigned __int64 *)sub_22077B0(32);
      v49 = v48;
      if ( v48 )
      {
        *v48 = 0;
        v48[1] = 0;
        v48[2] = 0;
        v48[3] = 0;
      }
      result = sub_2207820(4096);
      v50 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v49;
      *v49 = result;
      v49[3] = v50;
      v49[2] = 4096;
      v49[1] = 24;
      if ( result )
      {
LABEL_26:
        *(_DWORD *)(result + 8) = 2;
        *(_BYTE *)(result + 12) = 0;
        *(_DWORD *)(result + 16) = 13;
        *(_QWORD *)result = &unk_49E0E88;
        return result;
      }
      return result;
    case 'M':
      v16 = *(_QWORD **)(a1 + 16);
      result = (*v16 + v16[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v16[1] = result - *v16 + 24;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        if ( !result )
          return 0;
        goto LABEL_23;
      }
      v42 = (unsigned __int64 *)sub_22077B0(32);
      v43 = v42;
      if ( v42 )
      {
        *v42 = 0;
        v42[1] = 0;
        v42[2] = 0;
        v42[3] = 0;
      }
      result = sub_2207820(4096);
      v44 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v43;
      *v43 = result;
      v43[3] = v44;
      v43[2] = 4096;
      v43[1] = 24;
      if ( result )
      {
LABEL_23:
        *(_DWORD *)(result + 8) = 2;
        *(_BYTE *)(result + 12) = 0;
        *(_DWORD *)(result + 16) = 17;
        *(_QWORD *)result = &unk_49E0E88;
        return result;
      }
      return result;
    case 'N':
      v18 = *(_QWORD **)(a1 + 16);
      result = (*v18 + v18[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v18[1] = result - *v18 + 24;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        if ( !result )
          return 0;
        goto LABEL_29;
      }
      v51 = (unsigned __int64 *)sub_22077B0(32);
      v52 = v51;
      if ( v51 )
      {
        *v51 = 0;
        v51[1] = 0;
        v51[2] = 0;
        v51[3] = 0;
      }
      result = sub_2207820(4096);
      v53 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v52;
      *v52 = result;
      v52[3] = v53;
      v52[2] = 4096;
      v52[1] = 24;
      if ( result )
      {
LABEL_29:
        *(_DWORD *)(result + 8) = 2;
        *(_BYTE *)(result + 12) = 0;
        *(_DWORD *)(result + 16) = 18;
        *(_QWORD *)result = &unk_49E0E88;
        return result;
      }
      return result;
    case 'O':
      v19 = *(_QWORD **)(a1 + 16);
      result = (*v19 + v19[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v19[1] = result - *v19 + 24;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        if ( !result )
          return 0;
        goto LABEL_32;
      }
      v63 = (unsigned __int64 *)sub_22077B0(32);
      v64 = v63;
      if ( v63 )
      {
        *v63 = 0;
        v63[1] = 0;
        v63[2] = 0;
        v63[3] = 0;
      }
      result = sub_2207820(4096);
      v65 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v64;
      *v64 = result;
      v64[3] = v65;
      v64[2] = 4096;
      v64[1] = 24;
      if ( result )
      {
LABEL_32:
        *(_DWORD *)(result + 8) = 2;
        *(_BYTE *)(result + 12) = 0;
        *(_DWORD *)(result + 16) = 19;
        *(_QWORD *)result = &unk_49E0E88;
        return result;
      }
      return result;
    case 'X':
      v23 = *(_QWORD **)(a1 + 16);
      result = (*v23 + v23[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v23[1] = result - *v23 + 24;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        if ( !result )
          return 0;
        goto LABEL_43;
      }
      v66 = (unsigned __int64 *)sub_22077B0(32);
      v67 = v66;
      if ( v66 )
      {
        *v66 = 0;
        v66[1] = 0;
        v66[2] = 0;
        v66[3] = 0;
      }
      result = sub_2207820(4096);
      v68 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v67;
      *v67 = result;
      v67[3] = v68;
      v67[2] = 4096;
      v67[1] = 24;
      if ( result )
      {
LABEL_43:
        *(_DWORD *)(result + 8) = 2;
        *(_BYTE *)(result + 12) = 0;
        *(_DWORD *)(result + 16) = 0;
        *(_QWORD *)result = &unk_49E0E88;
        return result;
      }
      return result;
    case '_':
      if ( !v7 )
        goto LABEL_7;
      break;
    default:
LABEL_7:
      *(_BYTE *)(a1 + 8) = 1;
      return 0;
  }
  v21 = v4[1];
  a2[1] = (size_t)(v4 + 2);
  *a2 = v5 - 2;
  switch ( v21 )
  {
    case 'J':
      v35 = *(_QWORD **)(a1 + 16);
      result = (*v35 + v35[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v35[1] = result - *v35 + 24;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        if ( !result )
          return 0;
        goto LABEL_79;
      }
      v78 = (unsigned __int64 *)sub_22077B0(32);
      v79 = v78;
      if ( v78 )
      {
        *v78 = 0;
        v78[1] = 0;
        v78[2] = 0;
        v78[3] = 0;
      }
      result = sub_2207820(4096);
      v80 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v79;
      *v79 = result;
      v79[3] = v80;
      v79[2] = 4096;
      v79[1] = 24;
      if ( result )
      {
LABEL_79:
        *(_DWORD *)(result + 8) = 2;
        *(_BYTE *)(result + 12) = 0;
        *(_DWORD *)(result + 16) = 14;
        *(_QWORD *)result = &unk_49E0E88;
        return result;
      }
      return result;
    case 'K':
      v31 = *(_QWORD **)(a1 + 16);
      result = (*v31 + v31[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v31[1] = result - *v31 + 24;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        if ( !result )
          return 0;
        goto LABEL_67;
      }
      v75 = (unsigned __int64 *)sub_22077B0(32);
      v76 = v75;
      if ( v75 )
      {
        *v75 = 0;
        v75[1] = 0;
        v75[2] = 0;
        v75[3] = 0;
      }
      result = sub_2207820(4096);
      v77 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v76;
      *v76 = result;
      v76[3] = v77;
      v76[2] = 4096;
      v76[1] = 24;
      if ( result )
      {
LABEL_67:
        *(_DWORD *)(result + 8) = 2;
        *(_BYTE *)(result + 12) = 0;
        *(_DWORD *)(result + 16) = 15;
        *(_QWORD *)result = &unk_49E0E88;
        return result;
      }
      return result;
    case 'N':
      v33 = *(_QWORD **)(a1 + 16);
      result = (*v33 + v33[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v33[1] = result - *v33 + 24;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        if ( !result )
          return 0;
        goto LABEL_73;
      }
      v87 = (unsigned __int64 *)sub_22077B0(32);
      v88 = v87;
      if ( v87 )
      {
        *v87 = 0;
        v87[1] = 0;
        v87[2] = 0;
        v87[3] = 0;
      }
      result = sub_2207820(4096);
      v89 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v88;
      *v88 = result;
      v88[3] = v89;
      v88[2] = 4096;
      v88[1] = 24;
      if ( result )
      {
LABEL_73:
        *(_DWORD *)(result + 8) = 2;
        *(_BYTE *)(result + 12) = 0;
        *(_DWORD *)(result + 16) = 1;
        *(_QWORD *)result = &unk_49E0E88;
        return result;
      }
      return result;
    case 'P':
      v29 = *(_QWORD **)(a1 + 16);
      result = (*v29 + v29[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v29[1] = result - *v29 + 24;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        if ( !result )
          return 0;
        goto LABEL_61;
      }
      v84 = (unsigned __int64 *)sub_22077B0(32);
      v85 = v84;
      if ( v84 )
      {
        *v84 = 0;
        v84[1] = 0;
        v84[2] = 0;
        v84[3] = 0;
      }
      result = sub_2207820(4096);
      v86 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v85;
      *v85 = result;
      v85[3] = v86;
      v85[2] = 4096;
      v85[1] = 24;
      if ( result )
      {
LABEL_61:
        *(_DWORD *)(result + 8) = 2;
        *(_BYTE *)(result + 12) = 0;
        *(_DWORD *)(result + 16) = 21;
        *(_QWORD *)result = &unk_49E0E88;
        return result;
      }
      return result;
    case 'Q':
      v34 = *(_QWORD **)(a1 + 16);
      result = (*v34 + v34[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v34[1] = result - *v34 + 24;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        if ( !result )
          return 0;
        goto LABEL_76;
      }
      v99 = (unsigned __int64 *)sub_22077B0(32);
      v100 = v99;
      if ( v99 )
      {
        *v99 = 0;
        v99[1] = 0;
        v99[2] = 0;
        v99[3] = 0;
      }
      result = sub_2207820(4096);
      v101 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v100;
      *v100 = result;
      v100[3] = v101;
      v100[2] = 4096;
      v100[1] = 24;
      if ( result )
      {
LABEL_76:
        *(_DWORD *)(result + 8) = 2;
        *(_BYTE *)(result + 12) = 0;
        *(_DWORD *)(result + 16) = 5;
        *(_QWORD *)result = &unk_49E0E88;
        return result;
      }
      return result;
    case 'S':
      v30 = *(_QWORD **)(a1 + 16);
      result = (*v30 + v30[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v30[1] = result - *v30 + 24;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        if ( !result )
          return 0;
        goto LABEL_64;
      }
      v96 = (unsigned __int64 *)sub_22077B0(32);
      v97 = v96;
      if ( v96 )
      {
        *v96 = 0;
        v96[1] = 0;
        v96[2] = 0;
        v96[3] = 0;
      }
      result = sub_2207820(4096);
      v98 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v97;
      *v97 = result;
      v97[3] = v98;
      v97[2] = 4096;
      v97[1] = 24;
      if ( result )
      {
LABEL_64:
        *(_DWORD *)(result + 8) = 2;
        *(_BYTE *)(result + 12) = 0;
        *(_DWORD *)(result + 16) = 6;
        *(_QWORD *)result = &unk_49E0E88;
        return result;
      }
      return result;
    case 'T':
      v32 = *(_QWORD **)(a1 + 16);
      result = (*v32 + v32[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v32[1] = result - *v32 + 24;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        if ( !result )
          return 0;
        goto LABEL_70;
      }
      v93 = (unsigned __int64 *)sub_22077B0(32);
      v94 = v93;
      if ( v93 )
      {
        *v93 = 0;
        v93[1] = 0;
        v93[2] = 0;
        v93[3] = 0;
      }
      result = sub_2207820(4096);
      v95 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v94;
      *v94 = result;
      v94[3] = v95;
      v94[2] = 4096;
      v94[1] = 24;
      if ( result )
      {
LABEL_70:
        *(_DWORD *)(result + 8) = 2;
        *(_BYTE *)(result + 12) = 0;
        *(_DWORD *)(result + 16) = 22;
        *(_QWORD *)result = &unk_49E0E88;
        return result;
      }
      return result;
    case 'U':
      v27 = *(_QWORD **)(a1 + 16);
      result = (*v27 + v27[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v27[1] = result - *v27 + 24;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        if ( !result )
          return 0;
        goto LABEL_55;
      }
      v81 = (unsigned __int64 *)sub_22077B0(32);
      v82 = v81;
      if ( v81 )
      {
        *v81 = 0;
        v81[1] = 0;
        v81[2] = 0;
        v81[3] = 0;
      }
      result = sub_2207820(4096);
      v83 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v82;
      *v82 = result;
      v82[3] = v83;
      v82[2] = 4096;
      v82[1] = 24;
      if ( result )
      {
LABEL_55:
        *(_DWORD *)(result + 8) = 2;
        *(_BYTE *)(result + 12) = 0;
        *(_DWORD *)(result + 16) = 7;
        *(_QWORD *)result = &unk_49E0E88;
        return result;
      }
      return result;
    case 'W':
      v28 = *(_QWORD **)(a1 + 16);
      result = (*v28 + v28[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v28[1] = result - *v28 + 24;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        if ( !result )
          return 0;
        goto LABEL_58;
      }
      v90 = (unsigned __int64 *)sub_22077B0(32);
      v91 = v90;
      if ( v90 )
      {
        *v90 = 0;
        v90[1] = 0;
        v90[2] = 0;
        v90[3] = 0;
      }
      result = sub_2207820(4096);
      v92 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v91;
      *v91 = result;
      v91[3] = v92;
      v91[2] = 4096;
      v91[1] = 24;
      if ( result )
      {
LABEL_58:
        *(_DWORD *)(result + 8) = 2;
        *(_BYTE *)(result + 12) = 0;
        *(_DWORD *)(result + 16) = 16;
        *(_QWORD *)result = &unk_49E0E88;
        return result;
      }
      break;
    default:
      goto LABEL_7;
  }
  return result;
}
