// Function: sub_34A76E0
// Address: 0x34a76e0
//
void __fastcall sub_34A76E0(__int64 a1, unsigned int a2, unsigned int *a3, __int64 a4)
{
  signed int v4; // esi
  unsigned int *v5; // r15
  __int64 v6; // r14
  unsigned int *v7; // r13
  unsigned int *i; // r14
  unsigned int v9; // r10d
  unsigned int v10; // edi
  __int64 v11; // rbx
  int v12; // edi
  unsigned int v13; // r9d
  char *v14; // rax
  __int64 v15; // rcx
  unsigned int v16; // esi
  char *v17; // rdx
  int v18; // r11d
  __int64 *v19; // r8
  __int64 v20; // r12
  char *v21; // rsi
  char *v22; // rsi
  _QWORD *v23; // r8
  unsigned int k; // esi
  __int64 v25; // r10
  _QWORD *v26; // rdx
  unsigned int v27; // edi
  unsigned int v28; // edx
  unsigned int v29; // edi
  char *v30; // r8
  __int64 v31; // r12
  _QWORD *v32; // rdx
  unsigned int v33; // r9d
  __int64 v34; // r11
  __int64 v35; // rax
  _QWORD *v36; // rsi
  char *v37; // rsi
  _QWORD *v38; // r8
  unsigned int j; // ecx
  __int64 v40; // r9
  char *v41; // rdx
  unsigned int v42; // edx
  unsigned int v43; // ecx
  unsigned int v44; // ebx
  unsigned int v45; // r14d
  __int64 *v46; // rax
  __int64 v47; // rsi
  unsigned int *v48; // r12
  unsigned int v49; // r11d
  unsigned int v50; // edi
  int v51; // r9d
  __int64 *v52; // rcx
  int v53; // r11d
  __int64 *v54; // r8
  __int64 v55; // r13
  __int64 *v56; // rdi
  __int64 *v57; // rdi
  _QWORD *v58; // r8
  unsigned int m; // edi
  __int64 v60; // r10
  __int64 *v61; // rcx
  unsigned int v62; // r14d
  unsigned int v63; // r9d
  __int64 *v64; // r8
  __int64 v65; // r13
  _QWORD *v66; // rcx
  unsigned int v67; // edx
  __int64 v68; // r10
  __int64 v69; // rax
  _QWORD *v70; // rdi
  __int64 *v71; // rsi
  _QWORD *v72; // rdi
  unsigned int v73; // ecx
  __int64 v74; // r8
  __int64 *v75; // rdx
  signed int v76; // [rsp+8h] [rbp-58h]
  __int64 *v77; // [rsp+8h] [rbp-58h]
  char *v78; // [rsp+10h] [rbp-50h]
  unsigned int *v79; // [rsp+10h] [rbp-50h]
  char **v83; // [rsp+30h] [rbp-30h]
  __int64 v84; // [rsp+30h] [rbp-30h]

  v4 = a2 - 1;
  v76 = v4;
  if ( !v4 )
    return;
  v5 = a3;
  v6 = v4;
  v83 = (char **)(a1 + 8LL * v4);
  v7 = &a3[v6];
  for ( i = (unsigned int *)(a4 + v6 * 4); ; --i )
  {
    v9 = *v7;
    v10 = *i;
    --v76;
    if ( *v7 != *i )
    {
      v11 = v76;
      if ( v76 == -1 )
        goto LABEL_18;
      do
      {
        v12 = v10 - v9;
        v13 = v5[v11];
        v14 = *v83;
        v15 = *(_QWORD *)(a1 + 8 * v11);
        if ( v12 <= 0 )
        {
          v28 = 11 - v13;
          if ( 11 - v13 > v9 )
            v28 = v9;
          v29 = -v12;
          if ( v28 <= v29 )
            v29 = v28;
          if ( v29 )
          {
            v78 = *v83;
            v30 = *v83;
            v31 = (__int64)&v14[v29 + 176];
            v32 = v14 + 176;
            v33 = v13 - ((_DWORD)v14 + 176);
            do
            {
              v34 = v33 + (unsigned int)v32;
              v35 = *(_QWORD *)v30;
              v32 = (_QWORD *)((char *)v32 + 1);
              v30 += 16;
              v36 = (_QWORD *)(v15 + 16 * v34);
              *v36 = v35;
              v36[1] = *((_QWORD *)v30 - 1);
              *(_BYTE *)(v15 + v34 + 176) = *((_BYTE *)v32 - 1);
            }
            while ( v32 != (_QWORD *)v31 );
            v14 = v78;
          }
          v37 = v14;
          v38 = v14 + 176;
          for ( j = v29; v9 != j; *((_BYTE *)v38 - 1) = v14[v40 + 176] )
          {
            v40 = j++;
            v37 += 16;
            v38 = (_QWORD *)((char *)v38 + 1);
            v41 = &v14[16 * v40];
            *((_QWORD *)v37 - 2) = *(_QWORD *)v41;
            *((_QWORD *)v37 - 1) = *((_QWORD *)v41 + 1);
          }
          v12 = -v29;
        }
        else
        {
          v16 = v9 - 1;
          if ( v12 > v13 )
            v12 = v5[v11];
          if ( v12 > 11 - v9 )
            v12 = 11 - v9;
          if ( v9 )
          {
            v17 = &v14[v16];
            v18 = v12 - (_DWORD)v14;
            v19 = (__int64 *)&v14[16 * v16];
            do
            {
              v20 = *v19;
              v19 -= 2;
              v21 = &v14[16 * (v18 + (_DWORD)v17)];
              *(_QWORD *)v21 = v20;
              *((_QWORD *)v21 + 1) = v19[3];
              v14[v18 + (_DWORD)v17 + 176] = v17[176];
              v22 = v17--;
            }
            while ( v22 != v14 );
          }
          v23 = v14 + 176;
          for ( k = v13 - v12; v13 != k; *((_BYTE *)v23 - 1) = *(_BYTE *)(v15 + v25 + 176) )
          {
            v25 = k++;
            v14 += 16;
            v23 = (_QWORD *)((char *)v23 + 1);
            v26 = (_QWORD *)(v15 + 16 * v25);
            *((_QWORD *)v14 - 2) = *v26;
            *((_QWORD *)v14 - 1) = v26[1];
          }
        }
        v5[v11] -= v12;
        v27 = *v7 + v12;
        *v7 = v27;
        v9 = v27;
        v10 = *i;
        if ( v9 >= *i )
          break;
        --v11;
      }
      while ( (_DWORD)v11 != -1 );
    }
    if ( !v76 )
      break;
LABEL_18:
    --v83;
    --v7;
  }
  if ( a2 > 1 )
  {
    v84 = 1;
    v79 = v5;
    do
    {
      v42 = *v5;
      v43 = *(_DWORD *)(a4 + 4 * v84 - 4);
      v44 = v84;
      if ( *v5 != v43 && (_DWORD)v84 != a2 )
      {
        do
        {
          v45 = v42 - v43;
          v46 = *(__int64 **)(a1 + 8LL * v44);
          v47 = *(_QWORD *)(a1 + 8 * v84 - 8);
          v48 = &v79[v44];
          v49 = *v48;
          if ( (int)(v42 - v43) <= 0 )
          {
            v62 = v43 - v42;
            if ( 11 - v42 <= v43 - v42 )
              v62 = 11 - v42;
            v63 = v62;
            if ( v49 <= v62 )
              v63 = *v48;
            if ( v63 )
            {
              v77 = *(__int64 **)(a1 + 8LL * v44);
              v64 = v77;
              v65 = (__int64)v46 + v63 + 176;
              v66 = v46 + 22;
              v67 = v42 - ((_DWORD)v46 + 176);
              do
              {
                v68 = v67 + (unsigned int)v66;
                v69 = *v64;
                v66 = (_QWORD *)((char *)v66 + 1);
                v64 += 2;
                v70 = (_QWORD *)(v47 + 16 * v68);
                *v70 = v69;
                v70[1] = *(v64 - 1);
                *(_BYTE *)(v47 + v68 + 176) = *((_BYTE *)v66 - 1);
              }
              while ( v66 != (_QWORD *)v65 );
              v46 = v77;
            }
            v71 = v46;
            v72 = v46 + 22;
            v73 = v63;
            if ( v49 > v62 )
            {
              do
              {
                v74 = v73++;
                v71 += 2;
                v72 = (_QWORD *)((char *)v72 + 1);
                v75 = &v46[2 * v74];
                *(v71 - 2) = *v75;
                *(v71 - 1) = v75[1];
                *((_BYTE *)v72 - 1) = *((_BYTE *)v46 + v74 + 176);
              }
              while ( v49 != v73 );
            }
            v51 = -v63;
          }
          else
          {
            v50 = v49 - 1;
            if ( v45 > v42 )
              v45 = v42;
            v51 = v45;
            if ( v45 > 11 - v49 )
              v51 = 11 - v49;
            if ( v49 )
            {
              v52 = (__int64 *)((char *)v46 + v50);
              v53 = v51 - (_DWORD)v46;
              v54 = &v46[2 * v50];
              do
              {
                v55 = *v54;
                v54 -= 2;
                v56 = &v46[2 * (unsigned int)(v53 + (_DWORD)v52)];
                *v56 = v55;
                v56[1] = v54[3];
                *((_BYTE *)v46 + (unsigned int)(v53 + (_DWORD)v52) + 176) = *((_BYTE *)v52 + 176);
                v57 = v52;
                v52 = (__int64 *)((char *)v52 - 1);
              }
              while ( v57 != v46 );
            }
            v58 = v46 + 22;
            for ( m = v42 - v51; m != v42; *((_BYTE *)v58 - 1) = *(_BYTE *)(v47 + v60 + 176) )
            {
              v60 = m++;
              v46 += 2;
              v58 = (_QWORD *)((char *)v58 + 1);
              v61 = (__int64 *)(v47 + 16 * v60);
              *(v46 - 2) = *v61;
              *(v46 - 1) = v61[1];
            }
          }
          *v48 += v51;
          v42 = *v5 - v51;
          *v5 = v42;
          v43 = *(_DWORD *)(a4 + 4 * v84 - 4);
          if ( v42 >= v43 )
            break;
          ++v44;
        }
        while ( a2 != v44 );
      }
      ++v84;
      ++v5;
    }
    while ( a2 != v84 );
  }
}
