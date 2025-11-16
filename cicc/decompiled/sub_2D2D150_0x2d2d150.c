// Function: sub_2D2D150
// Address: 0x2d2d150
//
void __fastcall sub_2D2D150(__int64 a1, unsigned int a2, unsigned int *a3, __int64 a4)
{
  signed int v4; // esi
  __int64 v6; // r14
  __int64 *v7; // r15
  unsigned int *v8; // r13
  unsigned int *i; // r14
  unsigned int v10; // edi
  unsigned int v11; // r11d
  __int64 v12; // rbx
  int v13; // r11d
  __int64 v14; // rax
  unsigned int v15; // r10d
  __int64 v16; // rdx
  unsigned int v17; // r8d
  __int64 v18; // rcx
  unsigned int v19; // r8d
  int *v20; // rsi
  __int64 v21; // rcx
  int v22; // edx
  __int64 v23; // rdi
  unsigned int v24; // esi
  __int64 v25; // rcx
  __int64 v26; // rdi
  unsigned int v27; // ecx
  unsigned int v28; // r11d
  __int64 v29; // rcx
  __int64 v30; // rsi
  int v31; // r12d
  unsigned int v32; // ecx
  __int64 j; // rdx
  __int64 v34; // rsi
  __int64 v35; // rbx
  unsigned int *v36; // r13
  __int64 v37; // r12
  unsigned int v38; // r8d
  unsigned int v39; // r10d
  unsigned int v40; // edx
  int v41; // r15d
  __int64 v42; // rax
  __int64 v43; // rdx
  unsigned int *v44; // r11
  unsigned int v45; // edi
  unsigned int v46; // r9d
  int v47; // ecx
  int v48; // r14d
  __int64 v49; // rcx
  unsigned int v50; // r9d
  int *v51; // rsi
  __int64 v52; // rcx
  int v53; // edx
  __int64 v54; // rdi
  unsigned int v55; // esi
  __int64 v56; // rcx
  __int64 v57; // rdi
  unsigned int v58; // r15d
  unsigned int v59; // r14d
  __int64 v60; // rcx
  __int64 v61; // rsi
  int v62; // edi
  unsigned int v63; // ecx
  __int64 v64; // rdx
  __int64 v65; // rsi
  signed int v66; // [rsp+0h] [rbp-50h]
  unsigned int *v70; // [rsp+18h] [rbp-38h]
  __int64 v71; // [rsp+20h] [rbp-30h]
  __int64 v72; // [rsp+20h] [rbp-30h]
  unsigned int v73; // [rsp+20h] [rbp-30h]

  v4 = a2 - 1;
  v66 = v4;
  if ( !v4 )
    return;
  v6 = v4;
  v7 = (__int64 *)(a1 + 8LL * v4);
  v8 = &a3[v6];
  for ( i = (unsigned int *)(a4 + v6 * 4); ; --i )
  {
    v10 = *v8;
    v11 = *i;
    --v66;
    if ( *v8 != *i )
    {
      v12 = v66;
      if ( v66 == -1 )
        goto LABEL_20;
      do
      {
        v13 = v11 - v10;
        v14 = *v7;
        v15 = a3[v12];
        v16 = *(_QWORD *)(a1 + 8 * v12);
        if ( v13 <= 0 )
        {
          v27 = 16 - v15;
          if ( 16 - v15 > v10 )
            v27 = v10;
          v28 = -v13;
          if ( v27 <= v28 )
            v28 = v27;
          v29 = 0;
          if ( v28 )
          {
            do
            {
              v30 = v15 + (unsigned int)v29;
              *(_DWORD *)(v16 + 8 * v30) = *(_DWORD *)(v14 + 8 * v29);
              *(_DWORD *)(v16 + 8 * v30 + 4) = *(_DWORD *)(v14 + 8 * v29 + 4);
              v31 = *(_DWORD *)(v14 + 4 * v29++ + 128);
              *(_DWORD *)(v16 + 4 * v30 + 128) = v31;
            }
            while ( v28 != v29 );
          }
          v32 = v28;
          for ( j = 0; v10 != v32; j += 4 )
          {
            v34 = v32++;
            *(_DWORD *)(v14 + 2 * j) = *(_DWORD *)(v14 + 8 * v34);
            *(_DWORD *)(v14 + 2 * j + 4) = *(_DWORD *)(v14 + 8 * v34 + 4);
            *(_DWORD *)(v14 + j + 128) = *(_DWORD *)(v14 + 4 * v34 + 128);
          }
          v13 = -v28;
        }
        else
        {
          v17 = v10 - 1;
          if ( v13 > v15 )
            v13 = a3[v12];
          if ( 16 - v10 <= v13 )
            v13 = 16 - v10;
          if ( v10 )
          {
            v71 = *(_QWORD *)(a1 + 8 * v12);
            v18 = v17;
            v19 = v13 + v17;
            v20 = (int *)(v14 + 8 * v18);
            v21 = v14 + 4 * v18 + 128;
            do
            {
              v22 = *v20;
              v23 = v19;
              v21 -= 4;
              v20 -= 2;
              --v19;
              *(_DWORD *)(v14 + 8 * v23) = v22;
              *(_DWORD *)(v14 + 8 * v23 + 4) = v20[3];
              *(_DWORD *)(v14 + 4 * v23 + 128) = *(_DWORD *)(v21 + 4);
            }
            while ( v14 + 124 != v21 );
            v16 = v71;
          }
          v24 = v15 - v13;
          if ( v15 != v15 - v13 )
          {
            v25 = 0;
            do
            {
              v26 = v24++;
              *(_DWORD *)(v14 + 2 * v25) = *(_DWORD *)(v16 + 8 * v26);
              *(_DWORD *)(v14 + 2 * v25 + 4) = *(_DWORD *)(v16 + 8 * v26 + 4);
              *(_DWORD *)(v14 + v25 + 128) = *(_DWORD *)(v16 + 4 * v26 + 128);
              v25 += 4;
            }
            while ( v15 != v24 );
          }
        }
        a3[v12] -= v13;
        v10 = v13 + *v8;
        *v8 = v10;
        v11 = *i;
        if ( v10 >= *i )
          break;
        --v12;
      }
      while ( (_DWORD)v12 != -1 );
    }
    if ( !v66 )
      break;
LABEL_20:
    --v8;
    --v7;
  }
  if ( a2 > 1 )
  {
    v35 = a1;
    v36 = a3;
    v37 = 1;
    v70 = a3;
    do
    {
      v38 = *v36;
      v39 = v37;
      v40 = *(_DWORD *)(a4 + 4 * v37 - 4);
      if ( *v36 != v40 && a2 != (_DWORD)v37 )
      {
        do
        {
          v41 = v38 - v40;
          v42 = *(_QWORD *)(v35 + 8LL * v39);
          v43 = *(_QWORD *)(v35 + 8 * v37 - 8);
          v44 = &v70[v39];
          v45 = *v44;
          if ( v41 <= 0 )
          {
            v58 = -v41;
            if ( 16 - v38 <= v58 )
              v58 = 16 - v38;
            v59 = v58;
            if ( v45 <= v58 )
              v59 = *v44;
            v60 = 0;
            if ( v59 )
            {
              v73 = *v44;
              do
              {
                v61 = v38 + (unsigned int)v60;
                *(_DWORD *)(v43 + 8 * v61) = *(_DWORD *)(v42 + 8 * v60);
                *(_DWORD *)(v43 + 8 * v61 + 4) = *(_DWORD *)(v42 + 8 * v60 + 4);
                v62 = *(_DWORD *)(v42 + 4 * v60++ + 128);
                *(_DWORD *)(v43 + 4 * v61 + 128) = v62;
              }
              while ( v59 != v60 );
              v45 = v73;
            }
            v63 = v59;
            v64 = 0;
            if ( v45 > v58 )
            {
              do
              {
                v65 = v63++;
                *(_DWORD *)(v42 + 2 * v64) = *(_DWORD *)(v42 + 8 * v65);
                *(_DWORD *)(v42 + 2 * v64 + 4) = *(_DWORD *)(v42 + 8 * v65 + 4);
                *(_DWORD *)(v42 + v64 + 128) = *(_DWORD *)(v42 + 4 * v65 + 128);
                v64 += 4;
              }
              while ( v45 != v63 );
            }
            v48 = -v59;
          }
          else
          {
            v46 = v45 - 1;
            if ( v41 > v38 )
              v41 = v38;
            v47 = 16 - v45;
            if ( v41 <= 16 - v45 )
              v47 = v41;
            v48 = v47;
            if ( v45 )
            {
              v72 = *(_QWORD *)(v35 + 8 * v37 - 8);
              v49 = v46;
              v50 = v48 + v46;
              v51 = (int *)(v42 + 8 * v49);
              v52 = v42 + 4 * v49 + 128;
              do
              {
                v53 = *v51;
                v54 = v50;
                v52 -= 4;
                v51 -= 2;
                --v50;
                *(_DWORD *)(v42 + 8 * v54) = v53;
                *(_DWORD *)(v42 + 8 * v54 + 4) = v51[3];
                *(_DWORD *)(v42 + 4 * v54 + 128) = *(_DWORD *)(v52 + 4);
              }
              while ( v42 + 124 != v52 );
              v43 = v72;
            }
            v55 = v38 - v48;
            if ( v38 - v48 != v38 )
            {
              v56 = 0;
              do
              {
                v57 = v55++;
                *(_DWORD *)(v42 + 2 * v56) = *(_DWORD *)(v43 + 8 * v57);
                *(_DWORD *)(v42 + 2 * v56 + 4) = *(_DWORD *)(v43 + 8 * v57 + 4);
                *(_DWORD *)(v42 + v56 + 128) = *(_DWORD *)(v43 + 4 * v57 + 128);
                v56 += 4;
              }
              while ( v55 != v38 );
            }
          }
          *v44 += v48;
          v38 = *v36 - v48;
          *v36 = v38;
          v40 = *(_DWORD *)(a4 + 4 * v37 - 4);
          if ( v38 >= v40 )
            break;
          ++v39;
        }
        while ( a2 != v39 );
      }
      ++v37;
      ++v36;
    }
    while ( a2 != v37 );
  }
}
