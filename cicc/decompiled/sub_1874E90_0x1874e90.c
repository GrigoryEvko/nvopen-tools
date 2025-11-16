// Function: sub_1874E90
// Address: 0x1874e90
//
void __fastcall sub_1874E90(__int64 *a1, __int64 *a2, __int64 a3)
{
  unsigned int v4; // r9d
  __int64 v5; // rdi
  unsigned int v6; // ecx
  __int64 *v7; // r10
  __int64 v8; // r11
  unsigned int v9; // r11d
  unsigned int v10; // edx
  __int64 *v11; // rsi
  __int64 v12; // r10
  __int64 v13; // r14
  __int64 *v14; // rbx
  unsigned int v15; // r8d
  __int64 v16; // rax
  __int64 v17; // rdx
  int v18; // esi
  __int64 *v19; // rsi
  int v20; // ecx
  int v21; // esi
  __int64 *v22; // rcx
  int v23; // edx
  __int64 *v24; // rbx
  unsigned int v25; // ecx
  __int64 v26; // rax
  unsigned int v27; // r10d
  __int64 *v28; // rdi
  __int64 v29; // r9
  unsigned int v30; // edi
  unsigned int v31; // r9d
  __int64 *v32; // rdx
  __int64 v33; // r8
  __int64 v34; // r12
  __int64 *v35; // r11
  unsigned int v36; // esi
  int v37; // ecx
  int v38; // ecx
  __int64 v39; // r8
  unsigned int v40; // edi
  __int64 *v41; // rdx
  __int64 v42; // rsi
  int v43; // eax
  int v44; // esi
  int v45; // esi
  __int64 v46; // r8
  __int64 v47; // rcx
  int v48; // edx
  __int64 *v49; // rax
  __int64 v50; // rdi
  __int64 *v51; // r10
  int v52; // ebx
  int v53; // ecx
  int v54; // ecx
  __int64 v55; // rdi
  __int64 *v56; // r9
  __int64 v57; // r13
  int v58; // r10d
  __int64 v59; // rsi
  int v60; // r13d
  int v61; // eax
  int v62; // ecx
  int v63; // ecx
  __int64 v64; // r8
  __int64 *v65; // r9
  int v66; // r13d
  __int64 v67; // rdi
  __int64 v68; // rsi
  int v69; // ebx
  int v70; // ebx
  int v71; // ebx
  int v72; // eax
  int v73; // r10d
  int v74; // ebx
  __int64 *v75; // r10
  int v76; // [rsp+8h] [rbp-78h]
  unsigned int v79; // [rsp+20h] [rbp-60h]
  __int64 *v80; // [rsp+20h] [rbp-60h]
  __int64 *v81; // [rsp+20h] [rbp-60h]
  __int64 *v82; // [rsp+28h] [rbp-58h]
  __int64 v83; // [rsp+38h] [rbp-48h] BYREF
  __int64 v84; // [rsp+40h] [rbp-40h] BYREF
  _QWORD v85[7]; // [rsp+48h] [rbp-38h] BYREF

  if ( a1 == a2 || a1 + 1 == a2 )
    return;
  v82 = a1 + 1;
  do
  {
    while ( 1 )
    {
      v15 = *(_DWORD *)(a3 + 24);
      v16 = *a1;
      v17 = *v82;
      v84 = *a1;
      v83 = v17;
      if ( !v15 )
      {
        ++*(_QWORD *)a3;
LABEL_13:
        v18 = 2 * v15;
        goto LABEL_14;
      }
      v4 = v15 - 1;
      v5 = *(_QWORD *)(a3 + 8);
      v6 = (v15 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v7 = (__int64 *)(v5 + 40LL * v6);
      v8 = *v7;
      if ( v17 == *v7 )
      {
LABEL_5:
        v9 = *((_DWORD *)v7 + 2);
        goto LABEL_6;
      }
      v71 = 1;
      v19 = 0;
      while ( v8 != -4 )
      {
        if ( !v19 && v8 == -8 )
          v19 = v7;
        v6 = v4 & (v71 + v6);
        v7 = (__int64 *)(v5 + 40LL * v6);
        v8 = *v7;
        if ( v17 == *v7 )
          goto LABEL_5;
        ++v71;
      }
      v72 = *(_DWORD *)(a3 + 16);
      if ( !v19 )
        v19 = v7;
      ++*(_QWORD *)a3;
      v20 = v72 + 1;
      if ( 4 * (v72 + 1) >= 3 * v15 )
        goto LABEL_13;
      if ( v15 - *(_DWORD *)(a3 + 20) - v20 > v15 >> 3 )
        goto LABEL_15;
      v18 = v15;
LABEL_14:
      sub_1874B30(a3, v18);
      sub_18721D0(a3, &v83, v85);
      v19 = (__int64 *)v85[0];
      v17 = v83;
      v20 = *(_DWORD *)(a3 + 16) + 1;
LABEL_15:
      *(_DWORD *)(a3 + 16) = v20;
      if ( *v19 != -4 )
        --*(_DWORD *)(a3 + 20);
      *v19 = v17;
      *((_DWORD *)v19 + 2) = 0;
      v19[2] = 0;
      v19[3] = 0;
      v19[4] = 0;
      v15 = *(_DWORD *)(a3 + 24);
      if ( !v15 )
      {
        ++*(_QWORD *)a3;
        goto LABEL_19;
      }
      v5 = *(_QWORD *)(a3 + 8);
      v16 = v84;
      v4 = v15 - 1;
      v9 = 0;
LABEL_6:
      v10 = v4 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v11 = (__int64 *)(v5 + 40LL * v10);
      v12 = *v11;
      if ( *v11 != v16 )
      {
        v69 = 1;
        v22 = 0;
        while ( v12 != -4 )
        {
          if ( !v22 && v12 == -8 )
            v22 = v11;
          v10 = v4 & (v69 + v10);
          v11 = (__int64 *)(v5 + 40LL * v10);
          v12 = *v11;
          if ( *v11 == v16 )
            goto LABEL_7;
          ++v69;
        }
        v70 = *(_DWORD *)(a3 + 16);
        if ( !v22 )
          v22 = v11;
        ++*(_QWORD *)a3;
        v23 = v70 + 1;
        if ( 4 * (v70 + 1) < 3 * v15 )
        {
          if ( v15 - (v23 + *(_DWORD *)(a3 + 20)) > v15 >> 3 )
            goto LABEL_21;
          v21 = v15;
LABEL_20:
          sub_1874B30(a3, v21);
          sub_18721D0(a3, &v84, v85);
          v22 = (__int64 *)v85[0];
          v16 = v84;
          v23 = *(_DWORD *)(a3 + 16) + 1;
LABEL_21:
          *(_DWORD *)(a3 + 16) = v23;
          if ( *v22 != -4 )
            --*(_DWORD *)(a3 + 20);
          *v22 = v16;
          *((_DWORD *)v22 + 2) = 0;
          v22[2] = 0;
          v22[3] = 0;
          v22[4] = 0;
          v13 = *v82;
          v15 = *(_DWORD *)(a3 + 24);
          break;
        }
LABEL_19:
        v21 = 2 * v15;
        goto LABEL_20;
      }
LABEL_7:
      v13 = *v82;
      if ( *((_DWORD *)v11 + 2) <= v9 )
        break;
      v14 = v82 + 1;
      if ( a1 != v82 )
        memmove(a1 + 1, a1, (char *)v82 - (char *)a1);
      ++v82;
      *a1 = v13;
      if ( a2 == v14 )
        return;
    }
    v24 = v82;
    v79 = ((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4);
    while ( 1 )
    {
      v34 = *(v24 - 1);
      v35 = v24;
      v36 = v15;
      if ( v15 )
      {
        v25 = v15 - 1;
        v26 = *(_QWORD *)(a3 + 8);
        v27 = (v15 - 1) & v79;
        v28 = (__int64 *)(v26 + 40LL * v27);
        v29 = *v28;
        if ( v13 == *v28 )
        {
LABEL_26:
          v30 = *((_DWORD *)v28 + 2);
          goto LABEL_27;
        }
        v60 = 1;
        v41 = 0;
        while ( v29 != -4 )
        {
          if ( v29 == -8 && !v41 )
            v41 = v28;
          v27 = v25 & (v60 + v27);
          v28 = (__int64 *)(v26 + 40LL * v27);
          v29 = *v28;
          if ( v13 == *v28 )
            goto LABEL_26;
          ++v60;
        }
        v61 = *(_DWORD *)(a3 + 16);
        if ( !v41 )
          v41 = v28;
        ++*(_QWORD *)a3;
        v43 = v61 + 1;
        if ( 4 * v43 < 3 * v15 )
        {
          if ( v15 - *(_DWORD *)(a3 + 20) - v43 > v15 >> 3 )
            goto LABEL_34;
          sub_1874B30(a3, v15);
          v62 = *(_DWORD *)(a3 + 24);
          if ( !v62 )
          {
LABEL_133:
            ++*(_DWORD *)(a3 + 16);
            BUG();
          }
          v63 = v62 - 1;
          v64 = *(_QWORD *)(a3 + 8);
          v65 = 0;
          v35 = v24;
          v66 = 1;
          LODWORD(v67) = v63 & v79;
          v41 = (__int64 *)(v64 + 40LL * (v63 & v79));
          v68 = *v41;
          v43 = *(_DWORD *)(a3 + 16) + 1;
          if ( *v41 == v13 )
            goto LABEL_34;
          while ( v68 != -4 )
          {
            if ( v68 == -8 && !v65 )
              v65 = v41;
            v67 = v63 & (unsigned int)(v67 + v66);
            v41 = (__int64 *)(v64 + 40 * v67);
            v68 = *v41;
            if ( v13 == *v41 )
              goto LABEL_34;
            ++v66;
          }
          goto LABEL_67;
        }
      }
      else
      {
        ++*(_QWORD *)a3;
      }
      sub_1874B30(a3, 2 * v15);
      v37 = *(_DWORD *)(a3 + 24);
      if ( !v37 )
        goto LABEL_133;
      v38 = v37 - 1;
      v39 = *(_QWORD *)(a3 + 8);
      v35 = v24;
      v40 = v38 & v79;
      v41 = (__int64 *)(v39 + 40LL * (v38 & v79));
      v42 = *v41;
      v43 = *(_DWORD *)(a3 + 16) + 1;
      if ( v13 == *v41 )
        goto LABEL_34;
      v73 = 1;
      v65 = 0;
      while ( v42 != -4 )
      {
        if ( v42 != -8 || v65 )
          v41 = v65;
        v40 = v38 & (v73 + v40);
        v42 = *(_QWORD *)(v39 + 40LL * v40);
        if ( v13 == v42 )
        {
          v41 = (__int64 *)(v39 + 40LL * v40);
          goto LABEL_34;
        }
        ++v73;
        v65 = v41;
        v41 = (__int64 *)(v39 + 40LL * v40);
      }
LABEL_67:
      if ( v65 )
        v41 = v65;
LABEL_34:
      *(_DWORD *)(a3 + 16) = v43;
      if ( *v41 != -4 )
        --*(_DWORD *)(a3 + 20);
      *v41 = v13;
      *((_DWORD *)v41 + 2) = 0;
      v41[2] = 0;
      v41[3] = 0;
      v41[4] = 0;
      v36 = *(_DWORD *)(a3 + 24);
      if ( !v36 )
      {
        ++*(_QWORD *)a3;
        goto LABEL_38;
      }
      v26 = *(_QWORD *)(a3 + 8);
      v25 = v36 - 1;
      v30 = 0;
LABEL_27:
      v31 = v25 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
      v32 = (__int64 *)(v26 + 40LL * v31);
      v33 = *v32;
      if ( *v32 != v34 )
        break;
LABEL_28:
      --v24;
      if ( *((_DWORD *)v32 + 2) <= v30 )
        goto LABEL_43;
      v24[1] = *v24;
      v15 = *(_DWORD *)(a3 + 24);
    }
    v76 = 1;
    v51 = 0;
    while ( v33 != -4 )
    {
      if ( !v51 && v33 == -8 )
        v51 = v32;
      v31 = v25 & (v76 + v31);
      v32 = (__int64 *)(v26 + 40LL * v31);
      v33 = *v32;
      if ( v34 == *v32 )
        goto LABEL_28;
      ++v76;
    }
    v52 = *(_DWORD *)(a3 + 16);
    v49 = v51;
    if ( !v51 )
      v49 = v32;
    ++*(_QWORD *)a3;
    v48 = v52 + 1;
    if ( 4 * (v52 + 1) >= 3 * v36 )
    {
LABEL_38:
      v80 = v35;
      sub_1874B30(a3, 2 * v36);
      v44 = *(_DWORD *)(a3 + 24);
      if ( !v44 )
        goto LABEL_134;
      v45 = v44 - 1;
      v46 = *(_QWORD *)(a3 + 8);
      v35 = v80;
      LODWORD(v47) = v45 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
      v48 = *(_DWORD *)(a3 + 16) + 1;
      v49 = (__int64 *)(v46 + 40LL * (unsigned int)v47);
      v50 = *v49;
      if ( v34 != *v49 )
      {
        v74 = 1;
        v75 = 0;
        while ( v50 != -4 )
        {
          if ( v50 == -8 && !v75 )
            v75 = v49;
          v47 = v45 & (unsigned int)(v47 + v74);
          v49 = (__int64 *)(v46 + 40 * v47);
          v50 = *v49;
          if ( v34 == *v49 )
            goto LABEL_40;
          ++v74;
        }
        if ( v75 )
          v49 = v75;
      }
    }
    else if ( v36 - (v48 + *(_DWORD *)(a3 + 20)) <= v36 >> 3 )
    {
      v81 = v35;
      sub_1874B30(a3, v36);
      v53 = *(_DWORD *)(a3 + 24);
      if ( v53 )
      {
        v54 = v53 - 1;
        v55 = *(_QWORD *)(a3 + 8);
        v56 = 0;
        LODWORD(v57) = v54 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
        v35 = v81;
        v58 = 1;
        v48 = *(_DWORD *)(a3 + 16) + 1;
        v49 = (__int64 *)(v55 + 40LL * (unsigned int)v57);
        v59 = *v49;
        if ( *v49 != v34 )
        {
          while ( v59 != -4 )
          {
            if ( v59 == -8 && !v56 )
              v56 = v49;
            v57 = v54 & (unsigned int)(v57 + v58);
            v49 = (__int64 *)(v55 + 40 * v57);
            v59 = *v49;
            if ( v34 == *v49 )
              goto LABEL_40;
            ++v58;
          }
          if ( v56 )
            v49 = v56;
        }
        goto LABEL_40;
      }
LABEL_134:
      ++*(_DWORD *)(a3 + 16);
      BUG();
    }
LABEL_40:
    *(_DWORD *)(a3 + 16) = v48;
    if ( *v49 != -4 )
      --*(_DWORD *)(a3 + 20);
    *v49 = v34;
    *((_DWORD *)v49 + 2) = 0;
    v49[2] = 0;
    v49[3] = 0;
    v49[4] = 0;
LABEL_43:
    *v35 = v13;
    ++v82;
  }
  while ( a2 != v82 );
}
