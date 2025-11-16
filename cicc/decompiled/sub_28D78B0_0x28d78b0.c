// Function: sub_28D78B0
// Address: 0x28d78b0
//
void __fastcall sub_28D78B0(__int64 *a1, __int64 *a2, __int64 a3)
{
  unsigned int v4; // edi
  __int64 v5; // r9
  int v6; // r12d
  __int64 *v7; // r10
  unsigned int v8; // esi
  __int64 *v9; // rax
  __int64 v10; // r11
  unsigned int v11; // r11d
  int v12; // r12d
  __int64 *v13; // rsi
  unsigned int v14; // ecx
  __int64 *v15; // rax
  __int64 v16; // r10
  __int64 v17; // r14
  __int64 *v18; // rbx
  unsigned int v19; // r8d
  __int64 v20; // rdx
  __int64 v21; // rcx
  int v22; // esi
  int v23; // edx
  int v24; // esi
  int v25; // eax
  __int64 *v26; // rbx
  unsigned int v27; // edx
  int v28; // r13d
  __int64 *v29; // rdi
  __int64 v30; // rcx
  unsigned int v31; // r10d
  __int64 *v32; // rax
  __int64 v33; // r9
  unsigned int v34; // edi
  unsigned int v35; // r13d
  unsigned int v36; // r9d
  __int64 *v37; // rax
  __int64 v38; // r8
  __int64 v39; // r12
  __int64 *v40; // r11
  unsigned int v41; // esi
  int v42; // edx
  int v43; // edx
  __int64 v44; // r8
  unsigned int v45; // esi
  int v46; // eax
  __int64 v47; // rcx
  int v48; // esi
  int v49; // esi
  __int64 v50; // r8
  unsigned int v51; // ecx
  int v52; // eax
  __int64 *v53; // rdx
  __int64 v54; // rdi
  int v55; // eax
  int v56; // edx
  int v57; // edx
  __int64 v58; // r8
  __int64 *v59; // r9
  int v60; // r13d
  unsigned int v61; // ecx
  __int64 v62; // rsi
  __int64 *v63; // r10
  int v64; // eax
  int v65; // ecx
  int v66; // ecx
  __int64 *v67; // r9
  int v68; // r10d
  unsigned int v69; // r13d
  __int64 v70; // rdi
  __int64 v71; // rsi
  int v72; // eax
  int v73; // eax
  int v74; // r10d
  int v75; // ebx
  __int64 *v76; // r10
  int v77; // [rsp+10h] [rbp-80h]
  __int64 v78; // [rsp+18h] [rbp-78h]
  unsigned int v81; // [rsp+30h] [rbp-60h]
  __int64 *v82; // [rsp+30h] [rbp-60h]
  __int64 *v83; // [rsp+30h] [rbp-60h]
  __int64 *v84; // [rsp+38h] [rbp-58h]
  __int64 v85; // [rsp+48h] [rbp-48h] BYREF
  __int64 v86; // [rsp+50h] [rbp-40h] BYREF
  _QWORD v87[7]; // [rsp+58h] [rbp-38h] BYREF

  if ( a1 == a2 || a2 == a1 + 1 )
    return;
  v84 = a1 + 1;
  v78 = a3 + 1360;
  do
  {
    while ( 1 )
    {
      v19 = *(_DWORD *)(a3 + 1384);
      v20 = *a1;
      v21 = *v84;
      v86 = *a1;
      v85 = v21;
      if ( !v19 )
      {
        ++*(_QWORD *)(a3 + 1360);
        v87[0] = 0;
LABEL_13:
        v22 = 2 * v19;
        goto LABEL_14;
      }
      v4 = v19 - 1;
      v5 = *(_QWORD *)(a3 + 1368);
      v6 = 1;
      v7 = 0;
      v8 = (v19 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v9 = (__int64 *)(v5 + 16LL * v8);
      v10 = *v9;
      if ( v21 == *v9 )
      {
LABEL_5:
        v11 = *((_DWORD *)v9 + 2);
        goto LABEL_6;
      }
      while ( v10 != -4096 )
      {
        if ( !v7 && v10 == -8192 )
          v7 = v9;
        v8 = v4 & (v6 + v8);
        v9 = (__int64 *)(v5 + 16LL * v8);
        v10 = *v9;
        if ( v21 == *v9 )
          goto LABEL_5;
        ++v6;
      }
      if ( !v7 )
        v7 = v9;
      v73 = *(_DWORD *)(a3 + 1376);
      ++*(_QWORD *)(a3 + 1360);
      v23 = v73 + 1;
      v87[0] = v7;
      if ( 4 * (v73 + 1) >= 3 * v19 )
        goto LABEL_13;
      if ( v19 - *(_DWORD *)(a3 + 1380) - v23 > v19 >> 3 )
        goto LABEL_15;
      v22 = v19;
LABEL_14:
      sub_CE3370(v78, v22);
      sub_28CD4F0(v78, &v85, v87);
      v21 = v85;
      v7 = (__int64 *)v87[0];
      v23 = *(_DWORD *)(a3 + 1376) + 1;
LABEL_15:
      *(_DWORD *)(a3 + 1376) = v23;
      if ( *v7 != -4096 )
        --*(_DWORD *)(a3 + 1380);
      *v7 = v21;
      *((_DWORD *)v7 + 2) = 0;
      v19 = *(_DWORD *)(a3 + 1384);
      if ( !v19 )
      {
        ++*(_QWORD *)(a3 + 1360);
        v87[0] = 0;
        goto LABEL_19;
      }
      v5 = *(_QWORD *)(a3 + 1368);
      v20 = v86;
      v4 = v19 - 1;
      v11 = 0;
LABEL_6:
      v12 = 1;
      v13 = 0;
      v14 = v4 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v15 = (__int64 *)(v5 + 16LL * v14);
      v16 = *v15;
      if ( v20 != *v15 )
      {
        while ( v16 != -4096 )
        {
          if ( !v13 && v16 == -8192 )
            v13 = v15;
          v14 = v4 & (v12 + v14);
          v15 = (__int64 *)(v5 + 16LL * v14);
          v16 = *v15;
          if ( v20 == *v15 )
            goto LABEL_7;
          ++v12;
        }
        if ( !v13 )
          v13 = v15;
        v72 = *(_DWORD *)(a3 + 1376);
        ++*(_QWORD *)(a3 + 1360);
        v25 = v72 + 1;
        v87[0] = v13;
        if ( 4 * v25 < 3 * v19 )
        {
          if ( v19 - (v25 + *(_DWORD *)(a3 + 1380)) > v19 >> 3 )
            goto LABEL_21;
          v24 = v19;
LABEL_20:
          sub_CE3370(v78, v24);
          sub_28CD4F0(v78, &v86, v87);
          v20 = v86;
          v13 = (__int64 *)v87[0];
          v25 = *(_DWORD *)(a3 + 1376) + 1;
LABEL_21:
          *(_DWORD *)(a3 + 1376) = v25;
          if ( *v13 != -4096 )
            --*(_DWORD *)(a3 + 1380);
          *v13 = v20;
          *((_DWORD *)v13 + 2) = 0;
          v19 = *(_DWORD *)(a3 + 1384);
          v17 = *v84;
          break;
        }
LABEL_19:
        v24 = 2 * v19;
        goto LABEL_20;
      }
LABEL_7:
      v17 = *v84;
      if ( v11 >= *((_DWORD *)v15 + 2) )
        break;
      v18 = v84 + 1;
      if ( a1 != v84 )
        memmove(a1 + 1, a1, (char *)v84 - (char *)a1);
      ++v84;
      *a1 = v17;
      if ( a2 == v18 )
        return;
    }
    v26 = v84;
    v81 = ((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4);
    while ( 1 )
    {
      v39 = *(v26 - 1);
      v40 = v26;
      v41 = v19;
      if ( v19 )
      {
        v27 = v19 - 1;
        v28 = 1;
        v29 = 0;
        v30 = *(_QWORD *)(a3 + 1368);
        v31 = (v19 - 1) & v81;
        v32 = (__int64 *)(v30 + 16LL * v31);
        v33 = *v32;
        if ( *v32 == v17 )
        {
LABEL_26:
          v34 = *((_DWORD *)v32 + 2);
          goto LABEL_27;
        }
        while ( v33 != -4096 )
        {
          if ( v33 == -8192 && !v29 )
            v29 = v32;
          v31 = v27 & (v28 + v31);
          v32 = (__int64 *)(v30 + 16LL * v31);
          v33 = *v32;
          if ( *v32 == v17 )
            goto LABEL_26;
          ++v28;
        }
        if ( !v29 )
          v29 = v32;
        v55 = *(_DWORD *)(a3 + 1376);
        ++*(_QWORD *)(a3 + 1360);
        v46 = v55 + 1;
        if ( 4 * v46 < 3 * v19 )
        {
          if ( v19 - *(_DWORD *)(a3 + 1380) - v46 > v19 >> 3 )
            goto LABEL_34;
          sub_CE3370(v78, v19);
          v56 = *(_DWORD *)(a3 + 1384);
          if ( !v56 )
          {
LABEL_130:
            ++*(_DWORD *)(a3 + 1376);
            BUG();
          }
          v57 = v56 - 1;
          v58 = *(_QWORD *)(a3 + 1368);
          v59 = 0;
          v40 = v26;
          v60 = 1;
          v61 = v57 & v81;
          v46 = *(_DWORD *)(a3 + 1376) + 1;
          v29 = (__int64 *)(v58 + 16LL * (v57 & v81));
          v62 = *v29;
          if ( *v29 == v17 )
            goto LABEL_34;
          while ( v62 != -4096 )
          {
            if ( v62 == -8192 && !v59 )
              v59 = v29;
            v61 = v57 & (v60 + v61);
            v29 = (__int64 *)(v58 + 16LL * v61);
            v62 = *v29;
            if ( *v29 == v17 )
              goto LABEL_34;
            ++v60;
          }
          goto LABEL_58;
        }
      }
      else
      {
        ++*(_QWORD *)(a3 + 1360);
      }
      sub_CE3370(v78, 2 * v19);
      v42 = *(_DWORD *)(a3 + 1384);
      if ( !v42 )
        goto LABEL_130;
      v43 = v42 - 1;
      v40 = v26;
      v44 = *(_QWORD *)(a3 + 1368);
      v45 = v43 & v81;
      v46 = *(_DWORD *)(a3 + 1376) + 1;
      v29 = (__int64 *)(v44 + 16LL * (v43 & v81));
      v47 = *v29;
      if ( *v29 == v17 )
        goto LABEL_34;
      v74 = 1;
      v59 = 0;
      while ( v47 != -4096 )
      {
        if ( v47 != -8192 || v59 )
          v29 = v59;
        v45 = v43 & (v74 + v45);
        v47 = *(_QWORD *)(v44 + 16LL * v45);
        if ( v47 == v17 )
        {
          v29 = (__int64 *)(v44 + 16LL * v45);
          goto LABEL_34;
        }
        ++v74;
        v59 = v29;
        v29 = (__int64 *)(v44 + 16LL * v45);
      }
LABEL_58:
      if ( v59 )
        v29 = v59;
LABEL_34:
      *(_DWORD *)(a3 + 1376) = v46;
      if ( *v29 != -4096 )
        --*(_DWORD *)(a3 + 1380);
      *v29 = v17;
      *((_DWORD *)v29 + 2) = 0;
      v41 = *(_DWORD *)(a3 + 1384);
      if ( !v41 )
      {
        ++*(_QWORD *)(a3 + 1360);
        goto LABEL_38;
      }
      v30 = *(_QWORD *)(a3 + 1368);
      v27 = v41 - 1;
      v34 = 0;
LABEL_27:
      v35 = ((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4);
      v36 = v35 & v27;
      v37 = (__int64 *)(v30 + 16LL * (v35 & v27));
      v38 = *v37;
      if ( *v37 != v39 )
        break;
LABEL_28:
      --v26;
      if ( v34 >= *((_DWORD *)v37 + 2) )
        goto LABEL_43;
      v26[1] = *v26;
      v19 = *(_DWORD *)(a3 + 1384);
    }
    v77 = 1;
    v63 = 0;
    while ( v38 != -4096 )
    {
      if ( v38 == -8192 && !v63 )
        v63 = v37;
      v36 = v27 & (v77 + v36);
      v37 = (__int64 *)(v30 + 16LL * v36);
      v38 = *v37;
      if ( v39 == *v37 )
        goto LABEL_28;
      ++v77;
    }
    v53 = v63;
    if ( !v63 )
      v53 = v37;
    v64 = *(_DWORD *)(a3 + 1376);
    ++*(_QWORD *)(a3 + 1360);
    v52 = v64 + 1;
    if ( 4 * v52 >= 3 * v41 )
    {
LABEL_38:
      v82 = v40;
      sub_CE3370(v78, 2 * v41);
      v48 = *(_DWORD *)(a3 + 1384);
      if ( !v48 )
        goto LABEL_131;
      v49 = v48 - 1;
      v50 = *(_QWORD *)(a3 + 1368);
      v40 = v82;
      v51 = v49 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
      v52 = *(_DWORD *)(a3 + 1376) + 1;
      v53 = (__int64 *)(v50 + 16LL * v51);
      v54 = *v53;
      if ( v39 != *v53 )
      {
        v75 = 1;
        v76 = 0;
        while ( v54 != -4096 )
        {
          if ( v54 == -8192 && !v76 )
            v76 = v53;
          v51 = v49 & (v75 + v51);
          v53 = (__int64 *)(v50 + 16LL * v51);
          v54 = *v53;
          if ( v39 == *v53 )
            goto LABEL_40;
          ++v75;
        }
        if ( v76 )
          v53 = v76;
      }
    }
    else if ( v41 - (v52 + *(_DWORD *)(a3 + 1380)) <= v41 >> 3 )
    {
      v83 = v40;
      sub_CE3370(v78, v41);
      v65 = *(_DWORD *)(a3 + 1384);
      if ( v65 )
      {
        v66 = v65 - 1;
        v67 = 0;
        v40 = v83;
        v68 = 1;
        v69 = v66 & v35;
        v70 = *(_QWORD *)(a3 + 1368);
        v52 = *(_DWORD *)(a3 + 1376) + 1;
        v53 = (__int64 *)(v70 + 16LL * v69);
        v71 = *v53;
        if ( *v53 != v39 )
        {
          while ( v71 != -4096 )
          {
            if ( v71 == -8192 && !v67 )
              v67 = v53;
            v69 = v66 & (v68 + v69);
            v53 = (__int64 *)(v70 + 16LL * v69);
            v71 = *v53;
            if ( v39 == *v53 )
              goto LABEL_40;
            ++v68;
          }
          if ( v67 )
            v53 = v67;
        }
        goto LABEL_40;
      }
LABEL_131:
      ++*(_DWORD *)(a3 + 1376);
      BUG();
    }
LABEL_40:
    *(_DWORD *)(a3 + 1376) = v52;
    if ( *v53 != -4096 )
      --*(_DWORD *)(a3 + 1380);
    *v53 = v39;
    *((_DWORD *)v53 + 2) = 0;
LABEL_43:
    *v40 = v17;
    ++v84;
  }
  while ( a2 != v84 );
}
