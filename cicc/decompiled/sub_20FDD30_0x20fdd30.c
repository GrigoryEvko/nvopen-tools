// Function: sub_20FDD30
// Address: 0x20fdd30
//
void __fastcall sub_20FDD30(__int64 a1, unsigned int a2, unsigned int *a3, __int64 a4)
{
  signed int v4; // esi
  __int64 v6; // r15
  unsigned int *v7; // r14
  unsigned int *i; // r15
  unsigned int v9; // edi
  unsigned int v10; // r11d
  __int64 v11; // r12
  int v12; // r11d
  unsigned int v13; // ecx
  __int64 v14; // r8
  __int64 v15; // rax
  unsigned int v16; // r9d
  __int64 v17; // rdx
  unsigned int v18; // r9d
  __int64 v19; // rdi
  __int64 *v20; // rsi
  __int64 v21; // r10
  __int64 v22; // rcx
  __int64 v23; // rdx
  _QWORD *v24; // rdx
  unsigned int v25; // edi
  __int64 v26; // rdx
  __int64 v27; // r9
  _QWORD *v28; // rsi
  unsigned int v29; // edx
  unsigned int v30; // r11d
  __int64 v31; // rdx
  unsigned int v32; // r10d
  __int64 v33; // r9
  _QWORD *v34; // rsi
  __int64 v35; // rsi
  unsigned int v36; // esi
  __int64 j; // rdx
  __int64 v38; // r8
  _QWORD *v39; // rcx
  unsigned int *v40; // r15
  __int64 v41; // r14
  unsigned int v42; // edx
  unsigned int v43; // r12d
  unsigned int v44; // ecx
  unsigned int v45; // ebx
  __int64 v46; // rax
  __int64 v47; // r8
  unsigned int *v48; // r13
  unsigned int v49; // r10d
  unsigned int v50; // r9d
  unsigned int v51; // ecx
  int v52; // r11d
  __int64 v53; // rcx
  unsigned int v54; // r9d
  __int64 v55; // rdi
  __int64 *v56; // rsi
  __int64 v57; // r10
  __int64 v58; // rdx
  __int64 v59; // rcx
  _QWORD *v60; // rcx
  unsigned int v61; // edi
  __int64 v62; // rcx
  __int64 v63; // r9
  _QWORD *v64; // rsi
  unsigned int v65; // ebx
  unsigned int v66; // r11d
  __int64 v67; // rcx
  unsigned int v68; // r9d
  __int64 v69; // rdi
  _QWORD *v70; // rsi
  __int64 v71; // rsi
  unsigned int v72; // esi
  __int64 v73; // rdx
  __int64 v74; // rdi
  _QWORD *v75; // rcx
  signed int v76; // [rsp+0h] [rbp-58h]
  unsigned int v79; // [rsp+18h] [rbp-40h]
  unsigned int *v80; // [rsp+18h] [rbp-40h]
  __int64 *v81; // [rsp+20h] [rbp-38h]
  unsigned int v82; // [rsp+20h] [rbp-38h]

  v4 = a2 - 1;
  v76 = v4;
  if ( !v4 )
    return;
  v6 = v4;
  v81 = (__int64 *)(a1 + 8LL * v4);
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
        goto LABEL_20;
      do
      {
        v12 = v10 - v9;
        v13 = a3[v11];
        v14 = *(_QWORD *)(a1 + 8 * v11);
        v15 = *v81;
        if ( v12 <= 0 )
        {
          v29 = 8 - v13;
          if ( 8 - v13 > v9 )
            v29 = v9;
          v30 = -v12;
          if ( v29 <= v30 )
            v30 = v29;
          v31 = 0;
          v32 = v13 + v30;
          if ( v30 )
          {
            do
            {
              v33 = v13++;
              v34 = (_QWORD *)(v14 + 16 * v33);
              *v34 = *(_QWORD *)(v15 + 2 * v31);
              v34[1] = *(_QWORD *)(v15 + 2 * v31 + 8);
              v35 = *(_QWORD *)(v15 + v31 + 128);
              v31 += 8;
              *(_QWORD *)(v14 + 8 * v33 + 128) = v35;
            }
            while ( v32 != v13 );
          }
          v36 = v30;
          for ( j = 0; v9 != v36; j += 8 )
          {
            v38 = v36++;
            v39 = (_QWORD *)(v15 + 16 * v38);
            *(_QWORD *)(v15 + 2 * j) = *v39;
            *(_QWORD *)(v15 + 2 * j + 8) = v39[1];
            *(_QWORD *)(v15 + j + 128) = *(_QWORD *)(v15 + 8 * v38 + 128);
          }
          v12 = -v30;
        }
        else
        {
          v16 = v9 - 1;
          if ( v12 > v13 )
            v12 = a3[v11];
          if ( 8 - v9 <= v12 )
            v12 = 8 - v9;
          if ( v9 )
          {
            v17 = v16;
            v79 = a3[v11];
            v18 = v12 + v16;
            v19 = v15 + 8 * v17 + 128;
            v20 = (__int64 *)(v15 + 16 * v17);
            do
            {
              v21 = v18;
              v22 = *v20;
              v19 -= 8;
              v20 -= 2;
              v23 = v18--;
              v24 = (_QWORD *)(v15 + 16 * v23);
              *v24 = v22;
              v24[1] = v20[3];
              *(_QWORD *)(v15 + 8 * v21 + 128) = *(_QWORD *)(v19 + 8);
            }
            while ( v19 != v15 + 120 );
            v13 = v79;
          }
          v25 = v13 - v12;
          if ( v13 != v13 - v12 )
          {
            v26 = 0;
            do
            {
              v27 = v25++;
              v28 = (_QWORD *)(v14 + 16 * v27);
              *(_QWORD *)(v15 + 2 * v26) = *v28;
              *(_QWORD *)(v15 + 2 * v26 + 8) = v28[1];
              *(_QWORD *)(v15 + v26 + 128) = *(_QWORD *)(v14 + 8 * v27 + 128);
              v26 += 8;
            }
            while ( v13 != v25 );
          }
        }
        a3[v11] -= v12;
        v9 = v12 + *v7;
        *v7 = v9;
        v10 = *i;
        if ( v9 >= *i )
          break;
        --v11;
      }
      while ( (_DWORD)v11 != -1 );
    }
    if ( !v76 )
      break;
LABEL_20:
    --v81;
    --v7;
  }
  if ( a2 > 1 )
  {
    v40 = a3;
    v41 = 1;
    v80 = a3;
    do
    {
      v42 = *v40;
      v43 = v41;
      v44 = *(_DWORD *)(a4 + 4 * v41 - 4);
      if ( *v40 != v44 && a2 != (_DWORD)v41 )
      {
        do
        {
          v45 = v42 - v44;
          v46 = *(_QWORD *)(a1 + 8LL * v43);
          v47 = *(_QWORD *)(a1 + 8 * v41 - 8);
          v48 = &v80[v43];
          v49 = *v48;
          if ( (int)(v42 - v44) <= 0 )
          {
            v65 = v44 - v42;
            if ( 8 - v42 <= v44 - v42 )
              v65 = 8 - v42;
            v66 = v65;
            if ( v49 <= v65 )
              v66 = *v48;
            v67 = 0;
            v68 = v66 + v42;
            if ( v66 )
            {
              do
              {
                v69 = v42++;
                v70 = (_QWORD *)(v47 + 16 * v69);
                *v70 = *(_QWORD *)(v46 + 2 * v67);
                v70[1] = *(_QWORD *)(v46 + 2 * v67 + 8);
                v71 = *(_QWORD *)(v46 + v67 + 128);
                v67 += 8;
                *(_QWORD *)(v47 + 8 * v69 + 128) = v71;
              }
              while ( v42 != v68 );
            }
            v72 = v66;
            v73 = 0;
            if ( v49 > v65 )
            {
              do
              {
                v74 = v72++;
                v75 = (_QWORD *)(v46 + 16 * v74);
                *(_QWORD *)(v46 + 2 * v73) = *v75;
                *(_QWORD *)(v46 + 2 * v73 + 8) = v75[1];
                *(_QWORD *)(v46 + v73 + 128) = *(_QWORD *)(v46 + 8 * v74 + 128);
                v73 += 8;
              }
              while ( v49 != v72 );
            }
            v52 = -v66;
          }
          else
          {
            v50 = v49 - 1;
            if ( v45 > v42 )
              v45 = v42;
            v51 = 8 - v49;
            if ( v45 <= 8 - v49 )
              v51 = v45;
            v52 = v51;
            if ( v49 )
            {
              v53 = v50;
              v82 = v42;
              v54 = v52 + v50;
              v55 = v46 + 8 * v53 + 128;
              v56 = (__int64 *)(v46 + 16 * v53);
              do
              {
                v57 = v54;
                v58 = *v56;
                v55 -= 8;
                v56 -= 2;
                v59 = v54--;
                v60 = (_QWORD *)(v46 + 16 * v59);
                *v60 = v58;
                v60[1] = v56[3];
                *(_QWORD *)(v46 + 8 * v57 + 128) = *(_QWORD *)(v55 + 8);
              }
              while ( v46 + 120 != v55 );
              v42 = v82;
            }
            v61 = v42 - v52;
            if ( v42 - v52 != v42 )
            {
              v62 = 0;
              do
              {
                v63 = v61++;
                v64 = (_QWORD *)(v47 + 16 * v63);
                *(_QWORD *)(v46 + 2 * v62) = *v64;
                *(_QWORD *)(v46 + 2 * v62 + 8) = v64[1];
                *(_QWORD *)(v46 + v62 + 128) = *(_QWORD *)(v47 + 8 * v63 + 128);
                v62 += 8;
              }
              while ( v61 != v42 );
            }
          }
          *v48 += v52;
          v42 = *v40 - v52;
          *v40 = v42;
          v44 = *(_DWORD *)(a4 + 4 * v41 - 4);
          if ( v42 >= v44 )
            break;
          ++v43;
        }
        while ( a2 != v43 );
      }
      ++v41;
      ++v40;
    }
    while ( a2 != v41 );
  }
}
