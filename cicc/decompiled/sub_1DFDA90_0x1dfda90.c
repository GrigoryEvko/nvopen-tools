// Function: sub_1DFDA90
// Address: 0x1dfda90
//
void __fastcall sub_1DFDA90(__int64 a1, size_t a2)
{
  __int64 v4; // r14
  __int64 v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rsi
  unsigned int v9; // ecx
  __int64 *v10; // rdx
  __int64 v11; // r8
  __int64 v12; // r14
  __int64 *v13; // rax
  __int64 v14; // rdx
  unsigned int v15; // ecx
  int v16; // r8d
  __int64 v17; // r9
  __int64 v18; // r15
  int v19; // eax
  __int64 v20; // rbx
  __int64 v21; // r11
  int **v22; // r15
  int **v23; // rcx
  __int64 *v24; // r8
  __int64 v25; // r13
  int **v26; // r12
  __int64 v27; // r9
  int v28; // esi
  unsigned int v29; // edi
  int **v30; // rdx
  int *v31; // r11
  int *v32; // rdx
  int *v33; // rax
  __int64 v34; // rdx
  int v35; // esi
  __int64 v36; // rbx
  __int64 v37; // rax
  size_t v38; // rdx
  char *v39; // r8
  const void *v40; // rsi
  unsigned int v41; // esi
  int v42; // eax
  unsigned int v43; // edx
  int *v44; // rcx
  __int64 v45; // rax
  __int64 *v46; // rax
  __int64 v47; // rdx
  int v48; // ecx
  int v49; // r8d
  int v50; // r9d
  __int64 *v51; // rax
  __int64 v52; // rdx
  int v53; // ecx
  int v54; // r8d
  int v55; // r9d
  int v56; // edx
  int v57; // r10d
  int *v58; // rdi
  int v59; // ecx
  int v60; // edx
  int v61; // r9d
  __int64 v62; // rax
  int *v63; // [rsp+8h] [rbp-88h]
  __int64 *v64; // [rsp+10h] [rbp-80h]
  int v65; // [rsp+10h] [rbp-80h]
  __int64 v66; // [rsp+10h] [rbp-80h]
  char *v67; // [rsp+10h] [rbp-80h]
  size_t n; // [rsp+18h] [rbp-78h]
  size_t na; // [rsp+18h] [rbp-78h]
  size_t nb; // [rsp+18h] [rbp-78h]
  int v71; // [rsp+2Ch] [rbp-64h] BYREF
  __int64 v72; // [rsp+30h] [rbp-60h] BYREF
  int *v73; // [rsp+38h] [rbp-58h] BYREF
  char *v74; // [rsp+40h] [rbp-50h] BYREF
  __int64 v75; // [rsp+48h] [rbp-48h]
  int v76; // [rsp+50h] [rbp-40h]

  v5 = **(_QWORD **)(a2 + 32);
  v4 = *(_QWORD *)(a1 + 16);
  v72 = v5;
  sub_1E06620(v4);
  v6 = *(_QWORD *)(v4 + 1312);
  v7 = *(unsigned int *)(v6 + 48);
  if ( (_DWORD)v7 )
  {
    v8 = *(_QWORD *)(v6 + 32);
    v9 = (v7 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v10 = (__int64 *)(v8 + 16LL * v9);
    v11 = *v10;
    if ( v5 == *v10 )
    {
LABEL_3:
      if ( v10 != (__int64 *)(v8 + 16 * v7) && v10[1] )
      {
        v12 = a1 + 112;
        v13 = sub_1DFD350(a1 + 112, &v72);
        v74 = 0;
        v18 = v13[1];
        v75 = 0;
        v19 = *(_DWORD *)(v18 + 40);
        v76 = v19;
        if ( v19 )
        {
          v36 = (unsigned int)(v19 + 63) >> 6;
          v37 = malloc(8 * v36);
          v38 = 8 * v36;
          v39 = (char *)v37;
          if ( !v37 )
          {
            if ( 8 * v36 || (v62 = malloc(1u), v38 = 0, v39 = 0, !v62) )
            {
              v67 = v39;
              nb = v38;
              sub_16BD1C0("Allocation failed", 1u);
              v38 = nb;
              v39 = v67;
            }
            else
            {
              v39 = (char *)v62;
            }
          }
          v40 = *(const void **)(v18 + 24);
          v74 = v39;
          v75 = v36;
          memcpy(v39, v40, v38);
        }
        v20 = *(_QWORD *)(v72 + 32);
        v21 = v72 + 24;
        for ( n = a1 + 56; v21 != v20; v20 = *(_QWORD *)(v20 + 8) )
        {
          if ( **(_WORD **)(v20 + 16) && **(_WORD **)(v20 + 16) != 45 )
            break;
          v41 = *(_DWORD *)(a1 + 80);
          v42 = *(_DWORD *)(*(_QWORD *)(v20 + 32) + 8LL);
          v71 = v42;
          if ( v41 )
          {
            v17 = *(_QWORD *)(a1 + 64);
            v43 = (v41 - 1) & (37 * v42);
            v44 = (int *)(v17 + 8LL * v43);
            v16 = *v44;
            if ( v42 == *v44 )
            {
              v15 = v44[1];
LABEL_26:
              v45 = 8LL * (v15 >> 6);
              v14 = ~(1LL << v15);
              goto LABEL_27;
            }
            v65 = 1;
            v58 = 0;
            while ( v16 != -1 )
            {
              if ( v16 != -2 || v58 )
                v44 = v58;
              v43 = (v41 - 1) & (v65 + v43);
              v63 = (int *)(v17 + 8LL * v43);
              v16 = *v63;
              if ( v42 == *v63 )
              {
                v16 = v17 + 8 * v43;
                v15 = v63[1];
                goto LABEL_26;
              }
              ++v65;
              v58 = v44;
              v44 = (int *)(v17 + 8LL * v43);
            }
            if ( !v58 )
              v58 = v44;
            v59 = *(_DWORD *)(a1 + 72);
            ++*(_QWORD *)(a1 + 56);
            v15 = v59 + 1;
            if ( 4 * v15 < 3 * v41 )
            {
              v16 = v41 >> 3;
              if ( v41 - *(_DWORD *)(a1 + 76) - v15 > v41 >> 3 )
                goto LABEL_44;
              v66 = v21;
              goto LABEL_49;
            }
          }
          else
          {
            ++*(_QWORD *)(a1 + 56);
          }
          v66 = v21;
          v41 *= 2;
LABEL_49:
          sub_1BFDD60(n, v41);
          sub_1BFD720(n, &v71, &v73);
          v58 = v73;
          v42 = v71;
          v21 = v66;
          v15 = *(_DWORD *)(a1 + 72) + 1;
LABEL_44:
          *(_DWORD *)(a1 + 72) = v15;
          if ( *v58 != -1 )
            --*(_DWORD *)(a1 + 76);
          *v58 = v42;
          v14 = -2;
          v45 = 0;
          v58[1] = 0;
LABEL_27:
          *(_QWORD *)&v74[v45] &= v14;
          if ( (*(_BYTE *)v20 & 4) == 0 )
          {
            while ( (*(_BYTE *)(v20 + 46) & 8) != 0 )
              v20 = *(_QWORD *)(v20 + 8);
          }
        }
        sub_1DF8070(v18 + 48, (__int64)&v74, v14, v15, v16, v17);
        v22 = *(int ***)(a2 + 32);
        v23 = *(int ***)(a2 + 40);
        if ( v22 != v23 )
        {
          na = a2;
          v24 = (__int64 *)&v73;
          v25 = a1;
          v26 = v23;
          do
          {
            v33 = *v22;
            v34 = *(_QWORD *)(v25 + 8);
            v73 = *v22;
            v35 = *(_DWORD *)(v34 + 256);
            if ( v35 )
            {
              v27 = *(_QWORD *)(v34 + 240);
              v28 = v35 - 1;
              v29 = v28 & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
              v30 = (int **)(v27 + 16LL * v29);
              v31 = *v30;
              if ( v33 == *v30 )
              {
LABEL_12:
                if ( v33 != (int *)v72 )
                {
                  v32 = v30[1];
                  if ( (int *)na == v32 || v33 == **((int ***)v32 + 4) )
                  {
                    v64 = v24;
                    v46 = sub_1DFD350(v12, v24);
                    sub_1DF8070(v46[1] + 24, (__int64)&v74, v47, v48, v49, v50);
                    v51 = sub_1DFD350(v12, v64);
                    sub_1DF8070(v51[1] + 48, (__int64)&v74, v52, v53, v54, v55);
                    v24 = v64;
                  }
                }
                goto LABEL_15;
              }
              v56 = 1;
              while ( v31 != (int *)-8LL )
              {
                v57 = v56 + 1;
                v29 = v28 & (v56 + v29);
                v30 = (int **)(v27 + 16LL * v29);
                v31 = *v30;
                if ( v33 == *v30 )
                  goto LABEL_12;
                v56 = v57;
              }
            }
            if ( v33 != (int *)v72 )
              BUG();
LABEL_15:
            ++v22;
          }
          while ( v26 != v22 );
        }
        _libc_free((unsigned __int64)v74);
      }
    }
    else
    {
      v60 = 1;
      while ( v11 != -8 )
      {
        v61 = v60 + 1;
        v9 = (v7 - 1) & (v60 + v9);
        v10 = (__int64 *)(v8 + 16LL * v9);
        v11 = *v10;
        if ( v5 == *v10 )
          goto LABEL_3;
        v60 = v61;
      }
    }
  }
}
