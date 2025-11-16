// Function: sub_1C612E0
// Address: 0x1c612e0
//
char __fastcall sub_1C612E0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v5; // r13
  unsigned __int64 v8; // rsi
  char result; // al
  _QWORD *v10; // rax
  _QWORD *v11; // r11
  _QWORD *v12; // r12
  _QWORD *v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rcx
  _QWORD *v16; // r12
  __int64 v17; // rcx
  __int64 v18; // rdx
  unsigned int v19; // r15d
  __int64 v20; // r11
  unsigned int v21; // ecx
  __int64 v22; // rdx
  unsigned int v23; // esi
  __int64 v24; // rax
  unsigned __int64 v25; // r8
  unsigned int v26; // r13d
  unsigned int v27; // esi
  __int64 v28; // rax
  unsigned __int64 v29; // r8
  unsigned int v30; // edx
  __int64 v31; // rbx
  __int64 v32; // r14
  __int64 v33; // r9
  unsigned int v34; // ecx
  unsigned __int64 *v35; // rdx
  unsigned __int64 v36; // r8
  int v37; // eax
  __int64 v38; // rax
  unsigned int v39; // esi
  unsigned __int64 *v40; // rdi
  int v41; // ecx
  __int64 v42; // rax
  int v43; // r11d
  int v44; // ecx
  __int64 v45; // rax
  int v46; // eax
  __int64 v47; // rdx
  __int64 v48; // r15
  int v49; // ebx
  __int64 v50; // r14
  unsigned __int64 v51; // r13
  __int64 v52; // r9
  unsigned int v53; // ecx
  unsigned __int64 *v54; // rax
  unsigned __int64 v55; // rdi
  __int64 v56; // rdx
  unsigned int v57; // esi
  int v58; // edi
  unsigned int v59; // ecx
  _QWORD *v60; // rax
  _QWORD *i; // rdx
  int v62; // r10d
  unsigned __int64 *v63; // rdi
  int v64; // eax
  int v65; // edx
  unsigned int v66; // eax
  unsigned __int64 *v67; // rdi
  int v68; // eax
  int v69; // r11d
  unsigned __int64 *v70; // r8
  int v71; // edi
  int v72; // r10d
  int v73; // eax
  int v74; // esi
  int v75; // edi
  __int64 v76; // r11
  unsigned int v77; // eax
  int v78; // r15d
  unsigned int v79; // eax
  __int64 v80; // [rsp+10h] [rbp-80h]
  _QWORD *v81; // [rsp+18h] [rbp-78h]
  _QWORD *v82; // [rsp+18h] [rbp-78h]
  _QWORD *v83; // [rsp+18h] [rbp-78h]
  _QWORD *v84; // [rsp+18h] [rbp-78h]
  _QWORD *v85; // [rsp+20h] [rbp-70h]
  __int64 v86; // [rsp+28h] [rbp-68h]
  __int64 v87; // [rsp+28h] [rbp-68h]
  __int64 v88; // [rsp+28h] [rbp-68h]
  __int64 v89; // [rsp+28h] [rbp-68h]
  unsigned __int64 v90; // [rsp+38h] [rbp-58h] BYREF
  __int64 v91; // [rsp+40h] [rbp-50h] BYREF
  __int64 v92; // [rsp+48h] [rbp-48h] BYREF
  __int64 v93; // [rsp+50h] [rbp-40h] BYREF
  unsigned __int64 *v94[7]; // [rsp+58h] [rbp-38h] BYREF

  v4 = a4;
  v5 = a3;
  v8 = *(_QWORD *)(a3 + 40);
  v90 = v8;
  if ( v8 != *(_QWORD *)(a4 + 40) )
    return sub_15CCEE0(a2, a3, a4);
  v10 = (_QWORD *)a1[35];
  v11 = a1 + 34;
  v12 = a1 + 34;
  if ( !v10 )
    goto LABEL_59;
  v13 = (_QWORD *)a1[35];
  do
  {
    while ( 1 )
    {
      v14 = v13[2];
      v15 = v13[3];
      if ( v13[4] >= v8 )
        break;
      v13 = (_QWORD *)v13[3];
      if ( !v15 )
        goto LABEL_9;
    }
    v12 = v13;
    v13 = (_QWORD *)v13[2];
  }
  while ( v14 );
LABEL_9:
  if ( v11 != v12 )
  {
    if ( v12[4] <= v8 )
      goto LABEL_11;
    v12 = a1 + 34;
  }
  do
  {
    if ( v10[4] < v8 )
    {
      v10 = (_QWORD *)v10[3];
    }
    else
    {
      v12 = v10;
      v10 = (_QWORD *)v10[2];
    }
  }
  while ( v10 );
  if ( v11 == v12 || v12[4] > v8 )
  {
LABEL_59:
    v94[0] = &v90;
    v45 = sub_1C56950(a1 + 33, v12, v94);
    v8 = v90;
    v11 = a1 + 34;
    v12 = (_QWORD *)v45;
  }
  ++v12[5];
  v87 = (__int64)(v12 + 5);
  v46 = *((_DWORD *)v12 + 14);
  if ( !v46 )
  {
    if ( !*((_DWORD *)v12 + 15) )
      goto LABEL_66;
    v47 = *((unsigned int *)v12 + 16);
    if ( (unsigned int)v47 > 0x40 )
    {
      v81 = v11;
      j___libc_free_0(v12[6]);
      v11 = v81;
      *((_DWORD *)v12 + 16) = 0;
LABEL_64:
      v12[6] = 0;
LABEL_65:
      v12[7] = 0;
      goto LABEL_66;
    }
LABEL_82:
    v60 = (_QWORD *)v12[6];
    for ( i = &v60[2 * v47]; i != v60; v60 += 2 )
      *v60 = -8;
    goto LABEL_65;
  }
  v59 = 4 * v46;
  v47 = *((unsigned int *)v12 + 16);
  if ( (unsigned int)(4 * v46) < 0x40 )
    v59 = 64;
  if ( v59 >= (unsigned int)v47 )
    goto LABEL_82;
  v77 = v46 - 1;
  if ( v77 )
  {
    _BitScanReverse(&v77, v77);
    v78 = 1 << (33 - (v77 ^ 0x1F));
    if ( v78 < 64 )
      v78 = 64;
    if ( (_DWORD)v47 == v78 )
    {
      v84 = v11;
      sub_1C57350(v87);
      v11 = v84;
      goto LABEL_66;
    }
  }
  else
  {
    v78 = 64;
  }
  v83 = v11;
  j___libc_free_0(v12[6]);
  v79 = sub_1C521A0(v78);
  v11 = v83;
  *((_DWORD *)v12 + 16) = v79;
  if ( !v79 )
    goto LABEL_64;
  v12[6] = sub_22077B0(16LL * v79);
  sub_1C57350(v87);
  v11 = v83;
LABEL_66:
  v48 = *(_QWORD *)(v8 + 48);
  if ( v48 != v8 + 40 )
  {
    v82 = a1;
    v49 = 0;
    v80 = v4;
    v50 = v5;
    v51 = v8 + 40;
    v85 = v11;
    while ( 1 )
    {
      v56 = v48 - 24;
      if ( !v48 )
        v56 = 0;
      ++v49;
      v93 = v56;
      v57 = *((_DWORD *)v12 + 16);
      if ( !v57 )
        break;
      v52 = v12[6];
      v53 = (v57 - 1) & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
      v54 = (unsigned __int64 *)(v52 + 16LL * v53);
      v55 = *v54;
      if ( v56 != *v54 )
      {
        v69 = 1;
        v70 = 0;
        while ( v55 != -8 )
        {
          if ( !v70 && v55 == -16 )
            v70 = v54;
          v53 = (v57 - 1) & (v69 + v53);
          v54 = (unsigned __int64 *)(v52 + 16LL * v53);
          v55 = *v54;
          if ( v56 == *v54 )
            goto LABEL_69;
          ++v69;
        }
        v71 = *((_DWORD *)v12 + 14);
        if ( v70 )
          v54 = v70;
        ++v12[5];
        v58 = v71 + 1;
        if ( 4 * v58 < 3 * v57 )
        {
          if ( v57 - *((_DWORD *)v12 + 15) - v58 <= v57 >> 3 )
          {
LABEL_75:
            sub_14672C0(v87, v57);
            sub_1463AD0(v87, &v93, v94);
            v54 = v94[0];
            v56 = v93;
            v58 = *((_DWORD *)v12 + 14) + 1;
          }
          *((_DWORD *)v12 + 14) = v58;
          if ( *v54 != -8 )
            --*((_DWORD *)v12 + 15);
          *v54 = v56;
          *((_DWORD *)v54 + 2) = 0;
          goto LABEL_69;
        }
LABEL_74:
        v57 *= 2;
        goto LABEL_75;
      }
LABEL_69:
      *((_DWORD *)v54 + 2) = v49;
      v48 = *(_QWORD *)(v48 + 8);
      if ( v51 == v48 )
      {
        v5 = v50;
        v11 = v85;
        a1 = v82;
        v4 = v80;
        goto LABEL_45;
      }
    }
    ++v12[5];
    goto LABEL_74;
  }
LABEL_45:
  v10 = (_QWORD *)a1[35];
  if ( !v10 )
  {
    v16 = v11;
    goto LABEL_47;
  }
LABEL_11:
  v16 = v11;
  do
  {
    while ( 1 )
    {
      v17 = v10[2];
      v18 = v10[3];
      if ( v10[4] >= v90 )
        break;
      v10 = (_QWORD *)v10[3];
      if ( !v18 )
        goto LABEL_15;
    }
    v16 = v10;
    v10 = (_QWORD *)v10[2];
  }
  while ( v17 );
LABEL_15:
  if ( v11 == v16 || v16[4] > v90 )
  {
LABEL_47:
    v94[0] = &v90;
    v42 = sub_1C56950(a1 + 33, v16, v94);
    v91 = v5;
    v92 = v4;
    v16 = (_QWORD *)v42;
    if ( v5 == v4 )
      return 0;
    goto LABEL_18;
  }
  v91 = v5;
  v92 = v4;
  if ( v5 == v4 )
    return 0;
LABEL_18:
  v19 = *((_DWORD *)v16 + 16);
  v20 = (__int64)(v16 + 5);
  if ( !v19 )
  {
    ++v16[5];
LABEL_116:
    v74 = 2 * v19;
LABEL_117:
    sub_14672C0((__int64)(v16 + 5), v74);
    sub_1463AD0((__int64)(v16 + 5), &v91, v94);
    v63 = v94[0];
    v5 = v91;
    v20 = (__int64)(v16 + 5);
    v65 = *((_DWORD *)v16 + 14) + 1;
    goto LABEL_91;
  }
  v21 = v19 - 1;
  v22 = v16[6];
  v23 = (v19 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v24 = v22 + 16LL * v23;
  v25 = *(_QWORD *)v24;
  if ( v5 == *(_QWORD *)v24 )
  {
LABEL_20:
    v26 = *(_DWORD *)(v24 + 8);
    goto LABEL_21;
  }
  v62 = 1;
  v63 = 0;
  while ( v25 != -8 )
  {
    if ( !v63 && v25 == -16 )
      v63 = (unsigned __int64 *)v24;
    v23 = v21 & (v62 + v23);
    v24 = v22 + 16LL * v23;
    v25 = *(_QWORD *)v24;
    if ( v5 == *(_QWORD *)v24 )
      goto LABEL_20;
    ++v62;
  }
  if ( !v63 )
    v63 = (unsigned __int64 *)v24;
  v64 = *((_DWORD *)v16 + 14);
  ++v16[5];
  v65 = v64 + 1;
  if ( 4 * (v64 + 1) >= 3 * v19 )
    goto LABEL_116;
  if ( v19 - *((_DWORD *)v16 + 15) - v65 <= v19 >> 3 )
  {
    v74 = v19;
    goto LABEL_117;
  }
LABEL_91:
  *((_DWORD *)v16 + 14) = v65;
  if ( *v63 != -8 )
    --*((_DWORD *)v16 + 15);
  *v63 = v5;
  *((_DWORD *)v63 + 2) = 0;
  v19 = *((_DWORD *)v16 + 16);
  if ( !v19 )
  {
    ++v16[5];
    v66 = 0;
    goto LABEL_95;
  }
  v22 = v16[6];
  v4 = v92;
  v21 = v19 - 1;
  v26 = 0;
LABEL_21:
  v27 = v21 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v28 = v22 + 16LL * v27;
  v29 = *(_QWORD *)v28;
  if ( *(_QWORD *)v28 != v4 )
  {
    v72 = 1;
    v67 = 0;
    while ( v29 != -8 )
    {
      if ( v29 == -16 && !v67 )
        v67 = (unsigned __int64 *)v28;
      v27 = v21 & (v72 + v27);
      v28 = v22 + 16LL * v27;
      v29 = *(_QWORD *)v28;
      if ( *(_QWORD *)v28 == v4 )
        goto LABEL_22;
      ++v72;
    }
    if ( !v67 )
      v67 = (unsigned __int64 *)v28;
    v73 = *((_DWORD *)v16 + 14);
    ++v16[5];
    v68 = v73 + 1;
    if ( 4 * v68 < 3 * v19 )
    {
      if ( v19 - (v68 + *((_DWORD *)v16 + 15)) <= v19 >> 3 )
      {
        v89 = v20;
        sub_14672C0(v20, v19);
        sub_1463AD0(v89, &v92, v94);
        v67 = v94[0];
        v4 = v92;
        v20 = v89;
        v68 = *((_DWORD *)v16 + 14) + 1;
      }
      goto LABEL_96;
    }
    v66 = v19;
    v19 = v26;
LABEL_95:
    v88 = v20;
    v26 = v19;
    sub_14672C0(v20, 2 * v66);
    sub_1463AD0(v88, &v92, v94);
    v67 = v94[0];
    v4 = v92;
    v20 = v88;
    v68 = *((_DWORD *)v16 + 14) + 1;
LABEL_96:
    *((_DWORD *)v16 + 14) = v68;
    if ( *v67 != -8 )
      --*((_DWORD *)v16 + 15);
    *v67 = v4;
    v30 = 0;
    *((_DWORD *)v67 + 2) = 0;
LABEL_23:
    if ( v30 >= v26 )
    {
      v31 = v91 + 24;
      v32 = *(_QWORD *)(v91 + 40) + 40LL;
      if ( v91 + 24 != v32 )
      {
        v86 = v20;
        while ( 1 )
        {
          v38 = v31 - 24;
          if ( !v31 )
            v38 = 0;
          v93 = v38;
          if ( v38 == v92 )
            return 1;
          v39 = *((_DWORD *)v16 + 16);
          if ( !v39 )
            break;
          v33 = v16[6];
          v34 = (v39 - 1) & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
          v35 = (unsigned __int64 *)(v33 + 16LL * v34);
          v36 = *v35;
          if ( v38 != *v35 )
          {
            v43 = 1;
            v40 = 0;
            while ( v36 != -8 )
            {
              if ( v40 || v36 != -16 )
                v35 = v40;
              v75 = v43 + 1;
              v34 = (v39 - 1) & (v43 + v34);
              v76 = v33 + 16LL * v34;
              v36 = *(_QWORD *)v76;
              if ( v38 == *(_QWORD *)v76 )
              {
                v37 = *(_DWORD *)(v76 + 8);
                goto LABEL_28;
              }
              v43 = v75;
              v40 = v35;
              v35 = (unsigned __int64 *)(v33 + 16LL * v34);
            }
            v44 = *((_DWORD *)v16 + 14);
            if ( !v40 )
              v40 = v35;
            ++v16[5];
            v41 = v44 + 1;
            if ( 4 * v41 < 3 * v39 )
            {
              if ( v39 - *((_DWORD *)v16 + 15) - v41 > v39 >> 3 )
                goto LABEL_37;
LABEL_36:
              sub_14672C0(v86, v39);
              sub_1463AD0(v86, &v93, v94);
              v40 = v94[0];
              v38 = v93;
              v41 = *((_DWORD *)v16 + 14) + 1;
LABEL_37:
              *((_DWORD *)v16 + 14) = v41;
              if ( *v40 != -8 )
                --*((_DWORD *)v16 + 15);
              *v40 = v38;
              v37 = 0;
              *((_DWORD *)v40 + 2) = 0;
              goto LABEL_28;
            }
LABEL_35:
            v39 *= 2;
            goto LABEL_36;
          }
          v37 = *((_DWORD *)v35 + 2);
LABEL_28:
          if ( v37 == v26 )
          {
            v31 = *(_QWORD *)(v31 + 8);
            if ( v31 != v32 )
              continue;
          }
          return 0;
        }
        ++v16[5];
        goto LABEL_35;
      }
    }
    return 0;
  }
LABEL_22:
  v30 = *(_DWORD *)(v28 + 8);
  result = 1;
  if ( v30 <= v26 )
    goto LABEL_23;
  return result;
}
