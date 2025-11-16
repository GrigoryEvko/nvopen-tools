// Function: sub_2E00C30
// Address: 0x2e00c30
//
void __fastcall sub_2E00C30(__int64 a1, int a2, unsigned int *a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r9
  unsigned int v7; // edi
  __int64 v8; // r8
  unsigned int *v9; // rax
  unsigned __int64 v10; // rdi
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rdx
  unsigned int *v15; // r13
  __int64 v16; // r15
  __int64 v17; // rbx
  __int64 *v18; // rdx
  unsigned int v19; // r12d
  __int64 v20; // r10
  unsigned int v21; // eax
  unsigned __int64 v22; // rdx
  __int64 v23; // r14
  unsigned int v24; // eax
  __int64 v25; // rsi
  __int64 v26; // rax
  unsigned __int64 v27; // r11
  char *v28; // rsi
  unsigned __int64 v29; // rdi
  char *v30; // r14
  char *v31; // rbx
  __int64 v32; // r9
  _DWORD *v33; // rdi
  int v34; // r11d
  unsigned int v35; // ecx
  int *v36; // rax
  int v37; // r8d
  _DWORD *v38; // rdx
  _BYTE *v39; // rsi
  _QWORD *v40; // rdi
  unsigned int v41; // esi
  int v42; // eax
  int v43; // ecx
  __int64 v44; // r9
  int v45; // edx
  unsigned int v46; // eax
  int v47; // r8d
  int v48; // eax
  _DWORD *v49; // rdx
  __int64 v50; // rbx
  unsigned __int64 v51; // rax
  _QWORD *v52; // rcx
  _QWORD *v53; // rdi
  int v54; // eax
  int v55; // eax
  int v56; // eax
  __int64 v57; // r9
  _DWORD *v58; // r10
  int v59; // r11d
  unsigned int v60; // ecx
  int v61; // r8d
  int v62; // r10d
  int v63; // r11d
  unsigned __int64 v64; // [rsp+8h] [rbp-B8h]
  __int64 v65; // [rsp+18h] [rbp-A8h]
  unsigned __int64 v67; // [rsp+28h] [rbp-98h]
  unsigned int v68; // [rsp+3Ch] [rbp-84h]
  unsigned int *v69; // [rsp+40h] [rbp-80h]
  unsigned int *v70; // [rsp+48h] [rbp-78h]
  __int64 v72; // [rsp+58h] [rbp-68h]
  __int64 v73; // [rsp+58h] [rbp-68h]
  unsigned __int64 v74; // [rsp+68h] [rbp-58h] BYREF
  unsigned __int64 v75; // [rsp+70h] [rbp-50h] BYREF
  char *v76; // [rsp+78h] [rbp-48h]
  char *v77; // [rsp+80h] [rbp-40h]

  v4 = *(unsigned int *)(a1 + 200);
  v5 = *(_QWORD *)(a1 + 184);
  if ( !(_DWORD)v4 )
    return;
  v6 = (unsigned int)(v4 - 1);
  v7 = v6 & (37 * a2);
  v8 = *(unsigned int *)(v5 + 32LL * v7);
  v65 = v5 + 32LL * v7;
  if ( a2 != (_DWORD)v8 )
  {
    v62 = 1;
    while ( (_DWORD)v8 != -1 )
    {
      v7 = v6 & (v62 + v7);
      v8 = *(unsigned int *)(v5 + 32LL * v7);
      v65 = v5 + 32LL * v7;
      if ( a2 == (_DWORD)v8 )
        goto LABEL_3;
      ++v62;
    }
    return;
  }
LABEL_3:
  if ( v65 == v5 + 32 * v4 )
    return;
  v77 = 0;
  v76 = 0;
  v9 = *(unsigned int **)(v65 + 8);
  v10 = *(_QWORD *)(v65 + 16);
  v75 = 0;
  v67 = v10;
  if ( v9 == (unsigned int *)v10 )
    goto LABEL_34;
  v69 = v9;
  v70 = &a3[a4];
  do
  {
    v11 = a1 + 136;
    v12 = *(_QWORD *)(a1 + 144);
    v68 = *v69;
    if ( v12 )
    {
      do
      {
        while ( 1 )
        {
          v13 = *(_QWORD *)(v12 + 16);
          v14 = *(_QWORD *)(v12 + 24);
          if ( *v69 <= *(_DWORD *)(v12 + 32) )
            break;
          v12 = *(_QWORD *)(v12 + 24);
          if ( !v14 )
            goto LABEL_11;
        }
        v11 = v12;
        v12 = *(_QWORD *)(v12 + 16);
      }
      while ( v13 );
LABEL_11:
      if ( v11 != a1 + 136 && v68 < *(_DWORD *)(v11 + 32) )
        v11 = a1 + 136;
    }
    if ( a3 == v70 )
      goto LABEL_32;
    v15 = a3;
    v16 = v11;
    while ( 1 )
    {
      v19 = *v15;
      v20 = *(_QWORD *)(a1 + 112);
      v21 = *v15 & 0x7FFFFFFF;
      v22 = *(unsigned int *)(v20 + 160);
      v23 = 8LL * v21;
      if ( v21 >= (unsigned int)v22 || (v17 = *(_QWORD *)(*(_QWORD *)(v20 + 152) + 8LL * v21)) == 0 )
      {
        v24 = v21 + 1;
        if ( (unsigned int)v22 < v24 )
        {
          v27 = v24;
          if ( v24 != v22 )
          {
            if ( v24 >= v22 )
            {
              v50 = *(_QWORD *)(v20 + 168);
              v51 = v24 - v22;
              if ( v27 > *(unsigned int *)(v20 + 164) )
              {
                v64 = v51;
                v73 = *(_QWORD *)(a1 + 112);
                sub_C8D5F0(v20 + 152, (const void *)(v20 + 168), v27, 8u, v8, v6);
                v20 = v73;
                v51 = v64;
                v22 = *(unsigned int *)(v73 + 160);
              }
              v25 = *(_QWORD *)(v20 + 152);
              v52 = (_QWORD *)(v25 + 8 * v22);
              v53 = &v52[v51];
              if ( v52 != v53 )
              {
                do
                  *v52++ = v50;
                while ( v53 != v52 );
                LODWORD(v22) = *(_DWORD *)(v20 + 160);
                v25 = *(_QWORD *)(v20 + 152);
              }
              *(_DWORD *)(v20 + 160) = v51 + v22;
              goto LABEL_23;
            }
            *(_DWORD *)(v20 + 160) = v24;
          }
        }
        v25 = *(_QWORD *)(v20 + 152);
LABEL_23:
        v72 = v20;
        v26 = sub_2E10F30(v19);
        *(_QWORD *)(v25 + v23) = v26;
        v17 = v26;
        sub_2E11E80(v72, v26);
      }
      v18 = (__int64 *)sub_2E09D00(v17, *(_QWORD *)(v16 + 40));
      if ( v18 != (__int64 *)(*(_QWORD *)v17 + 24LL * *(unsigned int *)(v17 + 8))
        && (*(_DWORD *)((*v18 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v18 >> 1) & 3) <= (*(_DWORD *)((*(_QWORD *)(v16 + 40) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                               | (unsigned int)(*(__int64 *)(v16 + 40) >> 1)
                                                                                               & 3) )
      {
        break;
      }
      if ( v70 == ++v15 )
        goto LABEL_32;
    }
    v28 = v76;
    v74 = __PAIR64__(v68, v19);
    if ( v76 == v77 )
    {
      sub_2DFFD30(&v75, v76, &v74);
    }
    else
    {
      if ( v76 )
      {
        *(_QWORD *)v76 = v74;
        v28 = v76;
      }
      v76 = v28 + 8;
    }
    *(_DWORD *)(v16 + 48) = v19;
LABEL_32:
    ++v69;
  }
  while ( (unsigned int *)v67 != v69 );
  v67 = *(_QWORD *)(v65 + 8);
LABEL_34:
  if ( v67 )
    j_j___libc_free_0(v67);
  *(_DWORD *)v65 = -2;
  v29 = v75;
  v30 = v76;
  --*(_DWORD *)(a1 + 192);
  ++*(_DWORD *)(a1 + 196);
  if ( v30 != (char *)v29 )
  {
    v31 = (char *)v29;
    while ( 1 )
    {
      v41 = *(_DWORD *)(a1 + 200);
      if ( !v41 )
        break;
      v32 = *(_QWORD *)(a1 + 184);
      v33 = 0;
      v34 = 1;
      v35 = (v41 - 1) & (37 * *(_DWORD *)v31);
      v36 = (int *)(v32 + 32LL * v35);
      v37 = *v36;
      if ( *(_DWORD *)v31 != *v36 )
      {
        while ( v37 != -1 )
        {
          if ( v37 == -2 && !v33 )
            v33 = v36;
          v35 = (v41 - 1) & (v34 + v35);
          v36 = (int *)(v32 + 32LL * v35);
          v37 = *v36;
          if ( *(_DWORD *)v31 == *v36 )
            goto LABEL_39;
          ++v34;
        }
        if ( !v33 )
          v33 = v36;
        v54 = *(_DWORD *)(a1 + 192);
        ++*(_QWORD *)(a1 + 176);
        v45 = v54 + 1;
        if ( 4 * (v54 + 1) < 3 * v41 )
        {
          if ( v41 - *(_DWORD *)(a1 + 196) - v45 <= v41 >> 3 )
          {
            sub_2E00A10(a1 + 176, v41);
            v55 = *(_DWORD *)(a1 + 200);
            if ( !v55 )
            {
LABEL_94:
              ++*(_DWORD *)(a1 + 192);
              BUG();
            }
            v56 = v55 - 1;
            v57 = *(_QWORD *)(a1 + 184);
            v58 = 0;
            v59 = 1;
            v60 = v56 & (37 * *(_DWORD *)v31);
            v45 = *(_DWORD *)(a1 + 192) + 1;
            v33 = (_DWORD *)(v57 + 32LL * v60);
            v61 = *v33;
            if ( *(_DWORD *)v31 != *v33 )
            {
              while ( v61 != -1 )
              {
                if ( v61 == -2 && !v58 )
                  v58 = v33;
                v60 = v56 & (v59 + v60);
                v33 = (_DWORD *)(v57 + 32LL * v60);
                v61 = *v33;
                if ( *(_DWORD *)v31 == *v33 )
                  goto LABEL_47;
                ++v59;
              }
LABEL_74:
              if ( v58 )
                v33 = v58;
            }
          }
LABEL_47:
          *(_DWORD *)(a1 + 192) = v45;
          if ( *v33 != -1 )
            --*(_DWORD *)(a1 + 196);
          v48 = *(_DWORD *)v31;
          v40 = v33 + 2;
          *v40 = 0;
          v39 = 0;
          v40[1] = 0;
          *((_DWORD *)v40 - 2) = v48;
          v40[2] = 0;
          goto LABEL_50;
        }
LABEL_45:
        sub_2E00A10(a1 + 176, 2 * v41);
        v42 = *(_DWORD *)(a1 + 200);
        if ( !v42 )
          goto LABEL_94;
        v43 = v42 - 1;
        v44 = *(_QWORD *)(a1 + 184);
        v45 = *(_DWORD *)(a1 + 192) + 1;
        v46 = (v42 - 1) & (37 * *(_DWORD *)v31);
        v33 = (_DWORD *)(v44 + 32LL * (v43 & (unsigned int)(37 * *(_DWORD *)v31)));
        v47 = *v33;
        if ( *v33 != *(_DWORD *)v31 )
        {
          v63 = 1;
          v58 = 0;
          while ( v47 != -1 )
          {
            if ( !v58 && v47 == -2 )
              v58 = v33;
            v46 = v43 & (v63 + v46);
            v33 = (_DWORD *)(v44 + 32LL * v46);
            v47 = *v33;
            if ( *(_DWORD *)v31 == *v33 )
              goto LABEL_47;
            ++v63;
          }
          goto LABEL_74;
        }
        goto LABEL_47;
      }
LABEL_39:
      v38 = (_DWORD *)*((_QWORD *)v36 + 2);
      v39 = (_BYTE *)*((_QWORD *)v36 + 3);
      v40 = v36 + 2;
      if ( v38 == (_DWORD *)v39 )
      {
LABEL_50:
        v49 = v31 + 4;
        v31 += 8;
        sub_B8BBF0((__int64)v40, v39, v49);
        if ( v30 == v31 )
          goto LABEL_51;
      }
      else
      {
        if ( v38 )
        {
          *v38 = *((_DWORD *)v31 + 1);
          v38 = (_DWORD *)*((_QWORD *)v36 + 2);
        }
        v31 += 8;
        *((_QWORD *)v36 + 2) = v38 + 1;
        if ( v30 == v31 )
        {
LABEL_51:
          v29 = v75;
          goto LABEL_52;
        }
      }
    }
    ++*(_QWORD *)(a1 + 176);
    goto LABEL_45;
  }
LABEL_52:
  if ( v29 )
    j_j___libc_free_0(v29);
}
