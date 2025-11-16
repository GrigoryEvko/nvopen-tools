// Function: sub_1C29700
// Address: 0x1c29700
//
void __fastcall sub_1C29700(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v7; // r10
  __int64 v8; // rdx
  __int64 *v9; // r9
  int v10; // eax
  int v11; // edx
  __int64 v12; // rsi
  unsigned int v13; // eax
  __int64 *v14; // rcx
  __int64 v15; // rax
  __int64 *v16; // rbx
  __int64 *v17; // r8
  __int64 v18; // rdi
  unsigned int v19; // ecx
  _QWORD *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r12
  int v23; // eax
  int v24; // eax
  __int64 v25; // rsi
  int v26; // edi
  unsigned int v27; // r14d
  unsigned int v28; // ecx
  __int64 v29; // rdx
  unsigned int v30; // esi
  int v31; // r10d
  int v32; // r10d
  __int64 v33; // rsi
  unsigned int v34; // r14d
  int v35; // edx
  __int64 v36; // rcx
  int v37; // eax
  int v38; // edi
  unsigned int v39; // esi
  __int64 v40; // rdi
  unsigned int v41; // ebx
  unsigned int v42; // edx
  __int64 **v43; // rax
  __int64 *v44; // rcx
  int v45; // r12d
  __int64 **v46; // r14
  int v47; // eax
  int v48; // eax
  int v49; // edi
  int v50; // r10d
  int v51; // r10d
  __int64 v52; // rsi
  unsigned int v53; // r14d
  __int64 v54; // rcx
  _QWORD *v55; // rdi
  int v56; // edi
  int v57; // edi
  __int64 v58; // r12
  unsigned int v59; // edx
  __int64 *v60; // r8
  int v61; // esi
  __int64 **v62; // rcx
  int v63; // edi
  int v64; // edi
  __int64 v65; // r8
  __int64 **v66; // rdx
  unsigned int v67; // ebx
  int v68; // ecx
  __int64 *v69; // rsi
  int v70; // ebx
  __int64 v71; // rsi
  __int64 *v72; // [rsp+8h] [rbp-58h]
  __int64 *v73; // [rsp+8h] [rbp-58h]
  __int64 *v74; // [rsp+10h] [rbp-50h]
  int v75; // [rsp+10h] [rbp-50h]
  __int64 *v76; // [rsp+10h] [rbp-50h]
  __int64 v77; // [rsp+10h] [rbp-50h]
  __int64 v78; // [rsp+10h] [rbp-50h]
  __int64 v79; // [rsp+18h] [rbp-48h]
  _QWORD *v80; // [rsp+18h] [rbp-48h]
  __int64 v81; // [rsp+18h] [rbp-48h]
  int v82; // [rsp+18h] [rbp-48h]
  int v83; // [rsp+18h] [rbp-48h]
  __int64 v84; // [rsp+18h] [rbp-48h]
  __int64 v85; // [rsp+18h] [rbp-48h]
  __int64 v87; // [rsp+28h] [rbp-38h]
  __int64 *v88; // [rsp+28h] [rbp-38h]
  __int64 *v89; // [rsp+28h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 48);
  if ( v4 != a2 + 40 )
  {
    v7 = a2 + 40;
    do
    {
      if ( !v4 )
        BUG();
      if ( *(_BYTE *)(v4 - 8) == 78 )
      {
        v8 = *(_QWORD *)(v4 - 48);
        if ( !*(_BYTE *)(v8 + 16) && (*(_BYTE *)(v8 + 33) & 0x20) != 0 )
        {
          if ( (unsigned int)(*(_DWORD *)(v8 + 36) - 35) <= 3 )
            goto LABEL_30;
          if ( (*(_BYTE *)(v8 + 33) & 0x20) != 0 )
          {
            v37 = *(_DWORD *)(v8 + 36);
            if ( v37 == 4 || (unsigned int)(v37 - 116) <= 1 )
              goto LABEL_30;
          }
        }
      }
      v9 = (__int64 *)(v4 - 24);
      v10 = *(_DWORD *)(a1 + 80);
      if ( v10 )
      {
        v11 = v10 - 1;
        v12 = *(_QWORD *)(a1 + 64);
        v13 = (v10 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v14 = *(__int64 **)(v12 + 16LL * v13);
        if ( v9 == v14 )
          goto LABEL_9;
        v38 = 1;
        while ( v14 != (__int64 *)-8LL )
        {
          v13 = v11 & (v38 + v13);
          v14 = *(__int64 **)(v12 + 16LL * v13);
          if ( v9 == v14 )
            goto LABEL_9;
          ++v38;
        }
      }
      v39 = *(_DWORD *)(a4 + 24);
      if ( v39 )
      {
        v40 = *(_QWORD *)(a4 + 8);
        v41 = ((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4);
        v42 = (v39 - 1) & v41;
        v43 = (__int64 **)(v40 + 8LL * v42);
        v44 = *v43;
        if ( v9 == *v43 )
          goto LABEL_9;
        v45 = 1;
        v46 = 0;
        while ( v44 != (__int64 *)-8LL )
        {
          if ( v44 != (__int64 *)-16LL || v46 )
            v43 = v46;
          v42 = (v39 - 1) & (v45 + v42);
          v44 = *(__int64 **)(v40 + 8LL * v42);
          if ( v9 == v44 )
            goto LABEL_9;
          ++v45;
          v46 = v43;
          v43 = (__int64 **)(v40 + 8LL * v42);
        }
        if ( !v46 )
          v46 = v43;
        v47 = *(_DWORD *)(a4 + 16);
        ++*(_QWORD *)a4;
        v48 = v47 + 1;
        if ( 4 * v48 < 3 * v39 )
        {
          if ( v39 - *(_DWORD *)(a4 + 20) - v48 <= v39 >> 3 )
          {
            v78 = a3;
            v85 = v7;
            v89 = (__int64 *)(v4 - 24);
            sub_1353F00(a4, v39);
            v63 = *(_DWORD *)(a4 + 24);
            if ( !v63 )
            {
LABEL_113:
              ++*(_DWORD *)(a4 + 16);
              BUG();
            }
            v64 = v63 - 1;
            v65 = *(_QWORD *)(a4 + 8);
            v9 = (__int64 *)(v4 - 24);
            v66 = 0;
            v67 = v64 & v41;
            v7 = v85;
            a3 = v78;
            v68 = 1;
            v46 = (__int64 **)(v65 + 8LL * v67);
            v69 = *v46;
            v48 = *(_DWORD *)(a4 + 16) + 1;
            if ( v89 != *v46 )
            {
              while ( v69 != (__int64 *)-8LL )
              {
                if ( !v66 && v69 == (__int64 *)-16LL )
                  v66 = v46;
                v67 = v64 & (v67 + v68);
                v46 = (__int64 **)(v65 + 8LL * v67);
                v69 = *v46;
                if ( v89 == *v46 )
                  goto LABEL_48;
                ++v68;
              }
              if ( v66 )
                v46 = v66;
            }
          }
          goto LABEL_48;
        }
      }
      else
      {
        ++*(_QWORD *)a4;
      }
      v77 = a3;
      v84 = v7;
      v88 = (__int64 *)(v4 - 24);
      sub_1353F00(a4, 2 * v39);
      v56 = *(_DWORD *)(a4 + 24);
      if ( !v56 )
        goto LABEL_113;
      v9 = (__int64 *)(v4 - 24);
      v57 = v56 - 1;
      v58 = *(_QWORD *)(a4 + 8);
      v7 = v84;
      a3 = v77;
      v59 = v57 & (((unsigned int)v88 >> 9) ^ ((unsigned int)v88 >> 4));
      v46 = (__int64 **)(v58 + 8LL * v59);
      v60 = *v46;
      v48 = *(_DWORD *)(a4 + 16) + 1;
      if ( v88 != *v46 )
      {
        v61 = 1;
        v62 = 0;
        while ( v60 != (__int64 *)-8LL )
        {
          if ( v60 == (__int64 *)-16LL && !v62 )
            v62 = v46;
          v70 = v61 + 1;
          v71 = v57 & (v59 + v61);
          v46 = (__int64 **)(v58 + 8 * v71);
          v59 = v71;
          v60 = *v46;
          if ( v88 == *v46 )
            goto LABEL_48;
          v61 = v70;
        }
        if ( v62 )
          v46 = v62;
      }
LABEL_48:
      *(_DWORD *)(a4 + 16) = v48;
      if ( *v46 != (__int64 *)-8LL )
        --*(_DWORD *)(a4 + 20);
      *v46 = v9;
LABEL_9:
      v15 = 24LL * (*(_DWORD *)(v4 - 4) & 0xFFFFFFF);
      if ( (*(_BYTE *)(v4 - 1) & 0x40) != 0 )
      {
        v16 = *(__int64 **)(v4 - 32);
        v17 = &v16[(unsigned __int64)v15 / 8];
      }
      else
      {
        v17 = v9;
        v16 = &v9[v15 / 0xFFFFFFFFFFFFFFF8LL];
      }
      if ( v16 != v17 )
      {
        v87 = v7;
        while ( 1 )
        {
LABEL_16:
          v22 = *v16;
          if ( *(_BYTE *)(*v16 + 16) <= 0x17u )
            goto LABEL_15;
          v23 = *(_DWORD *)(a4 + 24);
          if ( !v23 )
            goto LABEL_15;
          v24 = v23 - 1;
          v25 = *(_QWORD *)(a4 + 8);
          v26 = 1;
          v27 = ((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4);
          v28 = v24 & v27;
          v29 = *(_QWORD *)(v25 + 8LL * (v24 & v27));
          if ( v22 == v29 )
            break;
          while ( v29 != -8 )
          {
            v28 = v24 & (v26 + v28);
            v29 = *(_QWORD *)(v25 + 8LL * v28);
            if ( v22 == v29 )
              goto LABEL_19;
            ++v26;
          }
          v16 += 3;
          if ( v17 == v16 )
          {
LABEL_36:
            v7 = v87;
            goto LABEL_30;
          }
        }
LABEL_19:
        v30 = *(_DWORD *)(a3 + 24);
        if ( !v30 )
        {
          ++*(_QWORD *)a3;
          goto LABEL_21;
        }
        v18 = *(_QWORD *)(a3 + 8);
        v19 = (v30 - 1) & v27;
        v20 = (_QWORD *)(v18 + 16LL * v19);
        v21 = *v20;
        if ( v22 == *v20 )
        {
LABEL_14:
          v20[1] = v9;
LABEL_15:
          v16 += 3;
          if ( v17 == v16 )
            goto LABEL_36;
          goto LABEL_16;
        }
        v75 = 1;
        v80 = 0;
        while ( v21 != -8 )
        {
          if ( !v80 )
          {
            if ( v21 != -16 )
              v20 = 0;
            v80 = v20;
          }
          v19 = (v30 - 1) & (v75 + v19);
          v20 = (_QWORD *)(v18 + 16LL * v19);
          v21 = *v20;
          if ( v22 == *v20 )
            goto LABEL_14;
          ++v75;
        }
        if ( v80 )
          v20 = v80;
        v49 = *(_DWORD *)(a3 + 16);
        ++*(_QWORD *)a3;
        v35 = v49 + 1;
        if ( 4 * (v49 + 1) >= 3 * v30 )
        {
LABEL_21:
          v79 = a3;
          v72 = v9;
          v74 = v17;
          sub_1C29540(a3, 2 * v30);
          a3 = v79;
          v31 = *(_DWORD *)(v79 + 24);
          if ( !v31 )
            goto LABEL_115;
          v32 = v31 - 1;
          v33 = *(_QWORD *)(v79 + 8);
          v34 = v32 & v27;
          v17 = v74;
          v9 = v72;
          v35 = *(_DWORD *)(v79 + 16) + 1;
          v20 = (_QWORD *)(v33 + 16LL * v34);
          v36 = *v20;
          if ( v22 != *v20 )
          {
            v83 = 1;
            v55 = 0;
            while ( v36 != -8 )
            {
              if ( !v55 && v36 == -16 )
                v55 = v20;
              v34 = v32 & (v83 + v34);
              v20 = (_QWORD *)(v33 + 16LL * v34);
              v36 = *v20;
              if ( v22 == *v20 )
                goto LABEL_23;
              ++v83;
            }
            goto LABEL_70;
          }
        }
        else if ( v30 - *(_DWORD *)(a3 + 20) - v35 <= v30 >> 3 )
        {
          v81 = a3;
          v73 = v9;
          v76 = v17;
          sub_1C29540(a3, v30);
          a3 = v81;
          v50 = *(_DWORD *)(v81 + 24);
          if ( !v50 )
          {
LABEL_115:
            ++*(_DWORD *)(a3 + 16);
            BUG();
          }
          v51 = v50 - 1;
          v52 = *(_QWORD *)(v81 + 8);
          v53 = v51 & v27;
          v17 = v76;
          v9 = v73;
          v35 = *(_DWORD *)(v81 + 16) + 1;
          v20 = (_QWORD *)(v52 + 16LL * v53);
          v54 = *v20;
          if ( v22 != *v20 )
          {
            v82 = 1;
            v55 = 0;
            while ( v54 != -8 )
            {
              if ( !v55 && v54 == -16 )
                v55 = v20;
              v53 = v51 & (v82 + v53);
              v20 = (_QWORD *)(v52 + 16LL * v53);
              v54 = *v20;
              if ( v22 == *v20 )
                goto LABEL_23;
              ++v82;
            }
LABEL_70:
            if ( v55 )
              v20 = v55;
          }
        }
LABEL_23:
        *(_DWORD *)(a3 + 16) = v35;
        if ( *v20 != -8 )
          --*(_DWORD *)(a3 + 20);
        *v20 = v22;
        v20[1] = 0;
        goto LABEL_14;
      }
LABEL_30:
      v4 = *(_QWORD *)(v4 + 8);
    }
    while ( v7 != v4 );
  }
}
