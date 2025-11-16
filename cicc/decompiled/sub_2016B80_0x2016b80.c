// Function: sub_2016B80
// Address: 0x2016b80
//
__int64 __fastcall sub_2016B80(__int64 a1, unsigned __int64 a2, __int64 a3, _DWORD *a4, _DWORD *a5)
{
  int v8; // r15d
  char v9; // dl
  __int64 v10; // rdi
  int v11; // esi
  unsigned int v12; // ecx
  _DWORD *v13; // r14
  int v14; // r8d
  char v15; // cl
  __int64 v16; // r9
  int v17; // esi
  int v18; // edi
  unsigned int v19; // r8d
  __int64 v20; // rdx
  int v21; // r10d
  char v22; // cl
  __int64 v23; // r9
  int v24; // esi
  int v25; // edi
  unsigned int v26; // r8d
  __int64 v27; // rdx
  int v28; // r10d
  __int64 result; // rax
  unsigned int v30; // esi
  unsigned int v31; // esi
  unsigned int v32; // esi
  unsigned int v33; // edi
  int v34; // r8d
  unsigned int v35; // r9d
  int v36; // ecx
  unsigned int v37; // edi
  int v38; // r8d
  unsigned int v39; // r9d
  int v40; // eax
  unsigned int v41; // ecx
  int v42; // edi
  unsigned int v43; // r9d
  int v44; // eax
  __int64 v45; // r11
  int v46; // r10d
  _DWORD *v47; // r9
  int v48; // eax
  __int64 v49; // r11
  __int64 v50; // r8
  int v51; // esi
  int v52; // ecx
  unsigned int v53; // edi
  int v54; // r9d
  __int64 v55; // r8
  int v56; // esi
  int v57; // ecx
  unsigned int v58; // edi
  int v59; // r9d
  __int64 v60; // rdi
  int v61; // edx
  unsigned int v62; // ecx
  int v63; // esi
  __int64 v64; // rdi
  int v65; // edx
  unsigned int v66; // ecx
  int v67; // esi
  int v68; // r9d
  _DWORD *v69; // r8
  __int64 v70; // r8
  int v71; // esi
  int v72; // ecx
  unsigned int v73; // edi
  int v74; // r9d
  int v75; // r11d
  __int64 v76; // r10
  __int64 v77; // r8
  int v78; // esi
  int v79; // ecx
  unsigned int v80; // edi
  int v81; // r9d
  int v82; // r11d
  __int64 v83; // r10
  int v84; // esi
  int v85; // edx
  int v86; // esi
  int v87; // esi
  int v88; // esi
  int v89; // edx
  int v90; // r11d
  int v91; // r9d
  int v92; // r11d

  v8 = sub_200F8F0(a1, a2, a3);
  v9 = *(_BYTE *)(a1 + 912) & 1;
  if ( v9 )
  {
    v10 = a1 + 920;
    v11 = 7;
  }
  else
  {
    v30 = *(_DWORD *)(a1 + 928);
    v10 = *(_QWORD *)(a1 + 920);
    if ( !v30 )
    {
      v41 = *(_DWORD *)(a1 + 912);
      ++*(_QWORD *)(a1 + 904);
      v13 = 0;
      v42 = (v41 >> 1) + 1;
LABEL_32:
      v43 = 3 * v30;
      goto LABEL_33;
    }
    v11 = v30 - 1;
  }
  v12 = v11 & (37 * v8);
  v13 = (_DWORD *)(v10 + 12LL * v12);
  v14 = *v13;
  if ( v8 == *v13 )
    goto LABEL_4;
  v46 = 1;
  v47 = 0;
  while ( v14 != -1 )
  {
    if ( !v47 && v14 == -2 )
      v47 = v13;
    v12 = v11 & (v46 + v12);
    v13 = (_DWORD *)(v10 + 12LL * v12);
    v14 = *v13;
    if ( v8 == *v13 )
      goto LABEL_4;
    ++v46;
  }
  v41 = *(_DWORD *)(a1 + 912);
  v30 = 8;
  if ( v47 )
    v13 = v47;
  ++*(_QWORD *)(a1 + 904);
  v43 = 24;
  v42 = (v41 >> 1) + 1;
  if ( !v9 )
  {
    v30 = *(_DWORD *)(a1 + 928);
    goto LABEL_32;
  }
LABEL_33:
  if ( 4 * v42 >= v43 )
  {
    sub_2015860(a1 + 904, 2 * v30);
    if ( (*(_BYTE *)(a1 + 912) & 1) != 0 )
    {
      v60 = a1 + 920;
      v61 = 7;
    }
    else
    {
      v85 = *(_DWORD *)(a1 + 928);
      v60 = *(_QWORD *)(a1 + 920);
      if ( !v85 )
        goto LABEL_154;
      v61 = v85 - 1;
    }
    v62 = v61 & (37 * v8);
    v13 = (_DWORD *)(v60 + 12LL * v62);
    v63 = *v13;
    if ( v8 != *v13 )
    {
      v91 = 1;
      v69 = 0;
      while ( v63 != -1 )
      {
        if ( !v69 && v63 == -2 )
          v69 = v13;
        v62 = v61 & (v91 + v62);
        v13 = (_DWORD *)(v60 + 12LL * v62);
        v63 = *v13;
        if ( v8 == *v13 )
          goto LABEL_67;
        ++v91;
      }
      goto LABEL_73;
    }
LABEL_67:
    v41 = *(_DWORD *)(a1 + 912);
    goto LABEL_35;
  }
  if ( v30 - *(_DWORD *)(a1 + 916) - v42 <= v30 >> 3 )
  {
    sub_2015860(a1 + 904, v30);
    if ( (*(_BYTE *)(a1 + 912) & 1) != 0 )
    {
      v64 = a1 + 920;
      v65 = 7;
      goto LABEL_70;
    }
    v89 = *(_DWORD *)(a1 + 928);
    v64 = *(_QWORD *)(a1 + 920);
    if ( v89 )
    {
      v65 = v89 - 1;
LABEL_70:
      v66 = v65 & (37 * v8);
      v13 = (_DWORD *)(v64 + 12LL * v66);
      v67 = *v13;
      if ( v8 != *v13 )
      {
        v68 = 1;
        v69 = 0;
        while ( v67 != -1 )
        {
          if ( v67 == -2 && !v69 )
            v69 = v13;
          v66 = v65 & (v68 + v66);
          v13 = (_DWORD *)(v64 + 12LL * v66);
          v67 = *v13;
          if ( v8 == *v13 )
            goto LABEL_67;
          ++v68;
        }
LABEL_73:
        if ( v69 )
          v13 = v69;
        goto LABEL_67;
      }
      goto LABEL_67;
    }
LABEL_154:
    *(_DWORD *)(a1 + 912) = (2 * (*(_DWORD *)(a1 + 912) >> 1) + 2) | *(_DWORD *)(a1 + 912) & 1;
    BUG();
  }
LABEL_35:
  *(_DWORD *)(a1 + 912) = (2 * (v41 >> 1) + 2) | v41 & 1;
  if ( *v13 != -1 )
    --*(_DWORD *)(a1 + 916);
  *v13 = v8;
  *(_QWORD *)(v13 + 1) = 0;
LABEL_4:
  sub_200D1B0(a1, v13 + 1);
  v15 = *(_BYTE *)(a1 + 352) & 1;
  if ( v15 )
  {
    v16 = a1 + 360;
    v17 = 7;
  }
  else
  {
    v32 = *(_DWORD *)(a1 + 368);
    v16 = *(_QWORD *)(a1 + 360);
    if ( !v32 )
    {
      v33 = *(_DWORD *)(a1 + 352);
      ++*(_QWORD *)(a1 + 344);
      v20 = 0;
      v34 = (v33 >> 1) + 1;
LABEL_18:
      v35 = 3 * v32;
      goto LABEL_19;
    }
    v17 = v32 - 1;
  }
  v18 = v13[1];
  v19 = v17 & (37 * v18);
  v20 = v16 + 24LL * v19;
  v21 = *(_DWORD *)v20;
  if ( v18 == *(_DWORD *)v20 )
    goto LABEL_7;
  v48 = 1;
  v49 = 0;
  while ( v21 != -1 )
  {
    if ( v21 == -2 && !v49 )
      v49 = v20;
    v19 = v17 & (v48 + v19);
    v20 = v16 + 24LL * v19;
    v21 = *(_DWORD *)v20;
    if ( v18 == *(_DWORD *)v20 )
      goto LABEL_7;
    ++v48;
  }
  v33 = *(_DWORD *)(a1 + 352);
  v35 = 24;
  v32 = 8;
  if ( v49 )
    v20 = v49;
  ++*(_QWORD *)(a1 + 344);
  v34 = (v33 >> 1) + 1;
  if ( !v15 )
  {
    v32 = *(_DWORD *)(a1 + 368);
    goto LABEL_18;
  }
LABEL_19:
  if ( 4 * v34 >= v35 )
  {
    sub_200F500(a1 + 344, 2 * v32);
    if ( (*(_BYTE *)(a1 + 352) & 1) != 0 )
    {
      v50 = a1 + 360;
      v51 = 7;
    }
    else
    {
      v86 = *(_DWORD *)(a1 + 368);
      v50 = *(_QWORD *)(a1 + 360);
      if ( !v86 )
        goto LABEL_152;
      v51 = v86 - 1;
    }
    v52 = v13[1];
    v53 = v51 & (37 * v52);
    v20 = v50 + 24LL * v53;
    v54 = *(_DWORD *)v20;
    if ( *(_DWORD *)v20 != v52 )
    {
      v92 = 1;
      v76 = 0;
      while ( v54 != -1 )
      {
        if ( !v76 && v54 == -2 )
          v76 = v20;
        v53 = v51 & (v92 + v53);
        v20 = v50 + 24LL * v53;
        v54 = *(_DWORD *)v20;
        if ( v52 == *(_DWORD *)v20 )
          goto LABEL_59;
        ++v92;
      }
      goto LABEL_80;
    }
LABEL_59:
    v33 = *(_DWORD *)(a1 + 352);
    goto LABEL_21;
  }
  if ( v32 - *(_DWORD *)(a1 + 356) - v34 <= v32 >> 3 )
  {
    sub_200F500(a1 + 344, v32);
    if ( (*(_BYTE *)(a1 + 352) & 1) != 0 )
    {
      v70 = a1 + 360;
      v71 = 7;
      goto LABEL_77;
    }
    v87 = *(_DWORD *)(a1 + 368);
    v70 = *(_QWORD *)(a1 + 360);
    if ( v87 )
    {
      v71 = v87 - 1;
LABEL_77:
      v72 = v13[1];
      v73 = v71 & (37 * v72);
      v20 = v70 + 24LL * v73;
      v74 = *(_DWORD *)v20;
      if ( *(_DWORD *)v20 != v72 )
      {
        v75 = 1;
        v76 = 0;
        while ( v74 != -1 )
        {
          if ( v74 == -2 && !v76 )
            v76 = v20;
          v73 = v71 & (v75 + v73);
          v20 = v70 + 24LL * v73;
          v74 = *(_DWORD *)v20;
          if ( v72 == *(_DWORD *)v20 )
            goto LABEL_59;
          ++v75;
        }
LABEL_80:
        if ( v76 )
          v20 = v76;
        goto LABEL_59;
      }
      goto LABEL_59;
    }
LABEL_152:
    *(_DWORD *)(a1 + 352) = (2 * (*(_DWORD *)(a1 + 352) >> 1) + 2) | *(_DWORD *)(a1 + 352) & 1;
    BUG();
  }
LABEL_21:
  *(_DWORD *)(a1 + 352) = (2 * (v33 >> 1) + 2) | v33 & 1;
  if ( *(_DWORD *)v20 != -1 )
    --*(_DWORD *)(a1 + 356);
  v36 = v13[1];
  *(_QWORD *)(v20 + 8) = 0;
  *(_DWORD *)(v20 + 16) = 0;
  *(_DWORD *)v20 = v36;
LABEL_7:
  *(_QWORD *)a4 = *(_QWORD *)(v20 + 8);
  a4[2] = *(_DWORD *)(v20 + 16);
  sub_200D1B0(a1, v13 + 2);
  v22 = *(_BYTE *)(a1 + 352) & 1;
  if ( v22 )
  {
    v23 = a1 + 360;
    v24 = 7;
  }
  else
  {
    v31 = *(_DWORD *)(a1 + 368);
    v23 = *(_QWORD *)(a1 + 360);
    if ( !v31 )
    {
      v37 = *(_DWORD *)(a1 + 352);
      ++*(_QWORD *)(a1 + 344);
      v27 = 0;
      v38 = (v37 >> 1) + 1;
LABEL_25:
      v39 = 3 * v31;
      goto LABEL_26;
    }
    v24 = v31 - 1;
  }
  v25 = v13[2];
  v26 = v24 & (37 * v25);
  v27 = v23 + 24LL * v26;
  v28 = *(_DWORD *)v27;
  if ( v25 == *(_DWORD *)v27 )
    goto LABEL_10;
  v44 = 1;
  v45 = 0;
  while ( v28 != -1 )
  {
    if ( !v45 && v28 == -2 )
      v45 = v27;
    v26 = v24 & (v44 + v26);
    v27 = v23 + 24LL * v26;
    v28 = *(_DWORD *)v27;
    if ( v25 == *(_DWORD *)v27 )
      goto LABEL_10;
    ++v44;
  }
  v37 = *(_DWORD *)(a1 + 352);
  v39 = 24;
  v31 = 8;
  if ( v45 )
    v27 = v45;
  ++*(_QWORD *)(a1 + 344);
  v38 = (v37 >> 1) + 1;
  if ( !v22 )
  {
    v31 = *(_DWORD *)(a1 + 368);
    goto LABEL_25;
  }
LABEL_26:
  if ( 4 * v38 >= v39 )
  {
    sub_200F500(a1 + 344, 2 * v31);
    if ( (*(_BYTE *)(a1 + 352) & 1) != 0 )
    {
      v55 = a1 + 360;
      v56 = 7;
    }
    else
    {
      v84 = *(_DWORD *)(a1 + 368);
      v55 = *(_QWORD *)(a1 + 360);
      if ( !v84 )
        goto LABEL_153;
      v56 = v84 - 1;
    }
    v57 = v13[2];
    v58 = v56 & (37 * v57);
    v27 = v55 + 24LL * v58;
    v59 = *(_DWORD *)v27;
    if ( *(_DWORD *)v27 != v57 )
    {
      v90 = 1;
      v83 = 0;
      while ( v59 != -1 )
      {
        if ( !v83 && v59 == -2 )
          v83 = v27;
        v58 = v56 & (v90 + v58);
        v27 = v55 + 24LL * v58;
        v59 = *(_DWORD *)v27;
        if ( v57 == *(_DWORD *)v27 )
          goto LABEL_63;
        ++v90;
      }
      goto LABEL_87;
    }
LABEL_63:
    v37 = *(_DWORD *)(a1 + 352);
    goto LABEL_28;
  }
  if ( v31 - *(_DWORD *)(a1 + 356) - v38 <= v31 >> 3 )
  {
    sub_200F500(a1 + 344, v31);
    if ( (*(_BYTE *)(a1 + 352) & 1) != 0 )
    {
      v77 = a1 + 360;
      v78 = 7;
      goto LABEL_84;
    }
    v88 = *(_DWORD *)(a1 + 368);
    v77 = *(_QWORD *)(a1 + 360);
    if ( v88 )
    {
      v78 = v88 - 1;
LABEL_84:
      v79 = v13[2];
      v80 = v78 & (37 * v79);
      v27 = v77 + 24LL * v80;
      v81 = *(_DWORD *)v27;
      if ( *(_DWORD *)v27 != v79 )
      {
        v82 = 1;
        v83 = 0;
        while ( v81 != -1 )
        {
          if ( v81 == -2 && !v83 )
            v83 = v27;
          v80 = v78 & (v82 + v80);
          v27 = v77 + 24LL * v80;
          v81 = *(_DWORD *)v27;
          if ( v79 == *(_DWORD *)v27 )
            goto LABEL_63;
          ++v82;
        }
LABEL_87:
        if ( v83 )
          v27 = v83;
        goto LABEL_63;
      }
      goto LABEL_63;
    }
LABEL_153:
    *(_DWORD *)(a1 + 352) = (2 * (*(_DWORD *)(a1 + 352) >> 1) + 2) | *(_DWORD *)(a1 + 352) & 1;
    BUG();
  }
LABEL_28:
  *(_DWORD *)(a1 + 352) = (2 * (v37 >> 1) + 2) | v37 & 1;
  if ( *(_DWORD *)v27 != -1 )
    --*(_DWORD *)(a1 + 356);
  v40 = v13[2];
  *(_QWORD *)(v27 + 8) = 0;
  *(_DWORD *)(v27 + 16) = 0;
  *(_DWORD *)v27 = v40;
LABEL_10:
  *(_QWORD *)a5 = *(_QWORD *)(v27 + 8);
  a5[2] = *(_DWORD *)(v27 + 16);
  result = *(unsigned int *)(a2 + 64);
  *(_DWORD *)(*(_QWORD *)a4 + 64LL) = result;
  *(_DWORD *)(*(_QWORD *)a5 + 64LL) = result;
  return result;
}
