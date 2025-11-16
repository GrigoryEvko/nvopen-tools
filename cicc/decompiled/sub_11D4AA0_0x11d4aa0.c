// Function: sub_11D4AA0
// Address: 0x11d4aa0
//
__int64 __fastcall sub_11D4AA0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r11
  _QWORD *v5; // r10
  __int64 v7; // r12
  __int64 v8; // rsi
  __int64 v9; // r8
  __int64 v10; // r9
  int v11; // r14d
  __int64 *v12; // rdx
  unsigned int v13; // edi
  __int64 *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // rax
  unsigned int v19; // eax
  _QWORD *v20; // rdi
  _QWORD *v21; // r9
  __int64 v22; // r10
  __int64 v23; // rdx
  __int64 v24; // r12
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // r15
  __int64 v28; // rbx
  __int64 v29; // r8
  __int64 *v30; // rax
  __int64 v31; // rcx
  __int64 v32; // rax
  _QWORD *v33; // rbx
  __int64 v34; // rax
  __int64 v35; // r15
  __int64 *v36; // rax
  __int64 v37; // rcx
  __int64 v38; // rdx
  __int64 v40; // rax
  __int64 v41; // rax
  unsigned __int64 v42; // rcx
  int v43; // eax
  int v44; // eax
  __int64 *v45; // rdx
  int v46; // eax
  int v47; // eax
  int v48; // esi
  unsigned int v49; // ecx
  __int64 v50; // rdi
  __int64 *v51; // r14
  int v52; // ecx
  int v53; // ecx
  __int64 v54; // rdi
  unsigned int v55; // r14d
  int v56; // ecx
  int v57; // ecx
  __int64 v58; // rdi
  int v59; // r13d
  int v60; // ecx
  int v61; // ecx
  __int64 v62; // rdi
  unsigned int v63; // r13d
  _QWORD *v64; // [rsp+8h] [rbp-108h]
  _QWORD *v65; // [rsp+8h] [rbp-108h]
  __int64 v66; // [rsp+8h] [rbp-108h]
  _QWORD *v67; // [rsp+8h] [rbp-108h]
  int v68; // [rsp+10h] [rbp-100h]
  __int64 v69; // [rsp+10h] [rbp-100h]
  int v70; // [rsp+10h] [rbp-100h]
  __int64 v71; // [rsp+10h] [rbp-100h]
  __int64 v72; // [rsp+10h] [rbp-100h]
  __int64 v73; // [rsp+10h] [rbp-100h]
  unsigned int v74; // [rsp+18h] [rbp-F8h]
  __int64 v75; // [rsp+18h] [rbp-F8h]
  int v76; // [rsp+18h] [rbp-F8h]
  __int64 v77; // [rsp+18h] [rbp-F8h]
  int v78; // [rsp+18h] [rbp-F8h]
  _QWORD *v79; // [rsp+18h] [rbp-F8h]
  __int64 v80; // [rsp+18h] [rbp-F8h]
  __int64 v81; // [rsp+20h] [rbp-F0h]
  __int64 v82; // [rsp+28h] [rbp-E8h]
  __int64 v83; // [rsp+28h] [rbp-E8h]
  __int64 v84; // [rsp+28h] [rbp-E8h]
  __int64 v85; // [rsp+28h] [rbp-E8h]
  _QWORD *v86; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v87; // [rsp+38h] [rbp-D8h]
  _QWORD v88[26]; // [rsp+40h] [rbp-D0h] BYREF

  v3 = a1;
  v5 = v88;
  v88[0] = a2;
  v7 = *(_QWORD *)(a2 + 40);
  v8 = *(unsigned int *)(a1 + 48);
  v87 = 0x1400000001LL;
  v86 = v88;
  v81 = a1 + 24;
  if ( !(_DWORD)v8 )
  {
    ++*(_QWORD *)(a1 + 24);
    goto LABEL_73;
  }
  v9 = *(_QWORD *)(a1 + 32);
  v10 = (unsigned int)(v8 - 1);
  v11 = 1;
  v12 = 0;
  v13 = v10 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v14 = (__int64 *)(v9 + 16LL * v13);
  v15 = *v14;
  if ( v7 != *v14 )
  {
    while ( v15 != -4096 )
    {
      if ( !v12 && v15 == -8192 )
        v12 = v14;
      v13 = v10 & (v11 + v13);
      v14 = (__int64 *)(v9 + 16LL * v13);
      v15 = *v14;
      if ( v7 == *v14 )
        goto LABEL_3;
      ++v11;
    }
    if ( !v12 )
      v12 = v14;
    v43 = *(_DWORD *)(v3 + 40);
    ++*(_QWORD *)(v3 + 24);
    v44 = v43 + 1;
    if ( 4 * v44 < (unsigned int)(3 * v8) )
    {
      if ( (int)v8 - *(_DWORD *)(v3 + 44) - v44 > (unsigned int)v8 >> 3 )
      {
LABEL_45:
        *(_DWORD *)(v3 + 40) = v44;
        if ( *v12 != -4096 )
          --*(_DWORD *)(v3 + 44);
        *v12 = v7;
        v16 = 0;
        v17 = 0;
        v12[1] = 0;
        goto LABEL_4;
      }
      v85 = v3;
      sub_11D3880(v81, v8);
      v3 = v85;
      v60 = *(_DWORD *)(v85 + 48);
      if ( v60 )
      {
        v61 = v60 - 1;
        v62 = *(_QWORD *)(v85 + 32);
        v9 = 0;
        v63 = v61 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v5 = v88;
        v10 = 1;
        v44 = *(_DWORD *)(v85 + 40) + 1;
        v12 = (__int64 *)(v62 + 16LL * v63);
        v8 = *v12;
        if ( v7 != *v12 )
        {
          while ( v8 != -4096 )
          {
            if ( !v9 && v8 == -8192 )
              v9 = (__int64)v12;
            v63 = v61 & (v10 + v63);
            v12 = (__int64 *)(v62 + 16LL * v63);
            v8 = *v12;
            if ( v7 == *v12 )
              goto LABEL_45;
            v10 = (unsigned int)(v10 + 1);
          }
          if ( v9 )
            v12 = (__int64 *)v9;
        }
        goto LABEL_45;
      }
LABEL_111:
      ++*(_DWORD *)(v3 + 40);
      BUG();
    }
LABEL_73:
    v84 = v3;
    sub_11D3880(v81, 2 * v8);
    v3 = v84;
    v56 = *(_DWORD *)(v84 + 48);
    if ( v56 )
    {
      v57 = v56 - 1;
      v9 = *(_QWORD *)(v84 + 32);
      v5 = v88;
      v8 = v57 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v44 = *(_DWORD *)(v84 + 40) + 1;
      v12 = (__int64 *)(v9 + 16 * v8);
      v58 = *v12;
      if ( v7 != *v12 )
      {
        v59 = 1;
        v10 = 0;
        while ( v58 != -4096 )
        {
          if ( !v10 && v58 == -8192 )
            v10 = (__int64)v12;
          v8 = v57 & (unsigned int)(v59 + v8);
          v12 = (__int64 *)(v9 + 16LL * (unsigned int)v8);
          v58 = *v12;
          if ( v7 == *v12 )
            goto LABEL_45;
          ++v59;
        }
        if ( v10 )
          v12 = (__int64 *)v10;
      }
      goto LABEL_45;
    }
    goto LABEL_111;
  }
LABEL_3:
  v16 = v14[1];
  v17 = v16;
LABEL_4:
  *(_QWORD *)(v16 + 56) = a2;
  v18 = *(unsigned int *)(a3 + 8);
  if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    v8 = a3 + 16;
    v83 = v3;
    sub_C8D5F0(a3, (const void *)(a3 + 16), v18 + 1, 8u, v9, v10);
    v18 = *(unsigned int *)(a3 + 8);
    v5 = v88;
    v3 = v83;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v18) = v17;
  v19 = v87;
  ++*(_DWORD *)(a3 + 8);
  v20 = v86;
  if ( !v19 )
    goto LABEL_32;
  v21 = v88;
  v22 = a3;
  do
  {
    v23 = v19--;
    v24 = v20[v23 - 1];
    LODWORD(v87) = v19;
    if ( (*(_DWORD *)(v24 + 4) & 0x7FFFFFF) != 0 )
    {
      v25 = 0;
      v82 = 8LL * (*(_DWORD *)(v24 + 4) & 0x7FFFFFF);
      while ( 1 )
      {
        v26 = *(_QWORD *)(v24 - 8);
        v8 = *(unsigned int *)(v3 + 48);
        v27 = *(_QWORD *)(v26 + 4 * v25);
        v28 = *(_QWORD *)(32LL * *(unsigned int *)(v24 + 72) + v26 + v25);
        if ( !(_DWORD)v8 )
          break;
        v29 = *(_QWORD *)(v3 + 32);
        v74 = (v8 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
        v30 = (__int64 *)(v29 + 16LL * v74);
        v31 = *v30;
        if ( v28 != *v30 )
        {
          v68 = 1;
          v45 = 0;
          while ( v31 != -4096 )
          {
            if ( !v45 && v31 == -8192 )
              v45 = v30;
            v74 = (v8 - 1) & (v74 + v68);
            v30 = (__int64 *)(v29 + 16LL * v74);
            v31 = *v30;
            if ( v28 == *v30 )
              goto LABEL_12;
            ++v68;
          }
          if ( !v45 )
            v45 = v30;
          v46 = *(_DWORD *)(v3 + 40);
          ++*(_QWORD *)(v3 + 24);
          v47 = v46 + 1;
          if ( 4 * v47 < (unsigned int)(3 * v8) )
          {
            if ( (int)v8 - *(_DWORD *)(v3 + 44) - v47 <= (unsigned int)v8 >> 3 )
            {
              v77 = v3;
              v65 = v21;
              v71 = v22;
              sub_11D3880(v81, v8);
              v3 = v77;
              v52 = *(_DWORD *)(v77 + 48);
              if ( !v52 )
                goto LABEL_111;
              v53 = v52 - 1;
              v54 = *(_QWORD *)(v77 + 32);
              v55 = v53 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
              v22 = v71;
              v21 = v65;
              v47 = *(_DWORD *)(v77 + 40) + 1;
              v45 = (__int64 *)(v54 + 16LL * v55);
              v8 = *v45;
              if ( v28 != *v45 )
              {
                v78 = 1;
                v29 = 0;
                while ( v8 != -4096 )
                {
                  if ( v8 == -8192 && !v29 )
                    v29 = (__int64)v45;
                  v55 = v53 & (v78 + v55);
                  v45 = (__int64 *)(v54 + 16LL * v55);
                  v8 = *v45;
                  if ( v28 == *v45 )
                    goto LABEL_54;
                  ++v78;
                }
                if ( v29 )
                  v45 = (__int64 *)v29;
              }
            }
            goto LABEL_54;
          }
LABEL_58:
          v75 = v3;
          v64 = v21;
          v69 = v22;
          sub_11D3880(v81, 2 * v8);
          v3 = v75;
          v48 = *(_DWORD *)(v75 + 48);
          if ( !v48 )
            goto LABEL_111;
          v8 = (unsigned int)(v48 - 1);
          v29 = *(_QWORD *)(v75 + 32);
          v22 = v69;
          v21 = v64;
          v49 = v8 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
          v47 = *(_DWORD *)(v75 + 40) + 1;
          v45 = (__int64 *)(v29 + 16LL * v49);
          v50 = *v45;
          if ( v28 != *v45 )
          {
            v76 = 1;
            v51 = 0;
            v70 = v8;
            while ( v50 != -4096 )
            {
              if ( !v51 && v50 == -8192 )
                v51 = v45;
              v49 = v70 & (v76 + v49);
              v8 = (unsigned int)(v76 + 1);
              v45 = (__int64 *)(v29 + 16LL * v49);
              v50 = *v45;
              if ( v28 == *v45 )
                goto LABEL_54;
              ++v76;
            }
            if ( v51 )
              v45 = v51;
          }
LABEL_54:
          *(_DWORD *)(v3 + 40) = v47;
          if ( *v45 != -4096 )
            --*(_DWORD *)(v3 + 44);
          *v45 = v28;
          v32 = 0;
          v45[1] = 0;
          goto LABEL_13;
        }
LABEL_12:
        v32 = v30[1];
LABEL_13:
        v33 = *(_QWORD **)(v32 + 16);
        v34 = v33[1];
        if ( v34 )
          goto LABEL_14;
        if ( *(_BYTE *)v27 != 84 || *v33 != *(_QWORD *)(v27 + 40) )
        {
LABEL_15:
          v35 = v22;
          if ( v86 != v21 )
            _libc_free(v86, v8);
          v36 = *(__int64 **)v35;
          v37 = *(_QWORD *)v35 + 8LL * *(unsigned int *)(v35 + 8);
          if ( *(_QWORD *)v35 != v37 )
          {
            do
            {
              v38 = *v36++;
              *(_QWORD *)(v38 + 56) = 0;
            }
            while ( (__int64 *)v37 != v36 );
          }
          *(_DWORD *)(v35 + 8) = 0;
          return 0;
        }
        v34 = v33[7];
        if ( v34 )
        {
LABEL_14:
          if ( v34 != v27 )
            goto LABEL_15;
        }
        else
        {
          v33[7] = v27;
          v40 = *(unsigned int *)(v22 + 8);
          if ( v40 + 1 > (unsigned __int64)*(unsigned int *)(v22 + 12) )
          {
            v8 = v22 + 16;
            v67 = v21;
            v73 = v3;
            v80 = v22;
            sub_C8D5F0(v22, (const void *)(v22 + 16), v40 + 1, 8u, v29, (__int64)v21);
            v22 = v80;
            v21 = v67;
            v3 = v73;
            v40 = *(unsigned int *)(v80 + 8);
          }
          *(_QWORD *)(*(_QWORD *)v22 + 8 * v40) = v33;
          v41 = (unsigned int)v87;
          v42 = HIDWORD(v87);
          ++*(_DWORD *)(v22 + 8);
          if ( v41 + 1 > v42 )
          {
            v8 = (__int64)v21;
            v66 = v22;
            v72 = v3;
            v79 = v21;
            sub_C8D5F0((__int64)&v86, v21, v41 + 1, 8u, v29, (__int64)v21);
            v41 = (unsigned int)v87;
            v22 = v66;
            v3 = v72;
            v21 = v79;
          }
          v86[v41] = v27;
          LODWORD(v87) = v87 + 1;
        }
        v25 += 8;
        if ( v82 == v25 )
        {
          v19 = v87;
          v20 = v86;
          goto LABEL_30;
        }
      }
      ++*(_QWORD *)(v3 + 24);
      goto LABEL_58;
    }
LABEL_30:
    ;
  }
  while ( v19 );
  v5 = v21;
LABEL_32:
  if ( v20 != v5 )
    _libc_free(v20, v8);
  return 1;
}
