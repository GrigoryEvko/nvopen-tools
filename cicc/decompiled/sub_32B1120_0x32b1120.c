// Function: sub_32B1120
// Address: 0x32b1120
//
void __fastcall sub_32B1120(__int64 a1, unsigned int a2, unsigned int *a3, __int64 a4)
{
  signed int v4; // esi
  __int64 v6; // r13
  __int64 **v7; // r14
  unsigned int *v8; // r12
  unsigned int *i; // r13
  unsigned int v10; // r9d
  unsigned int v11; // r8d
  __int64 v12; // rbx
  int v13; // r8d
  __int64 *v14; // rax
  unsigned int v15; // esi
  __int64 v16; // r11
  unsigned int v17; // edi
  __int64 v18; // rcx
  unsigned int v19; // edi
  __int64 *v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // r15
  __int64 *v23; // r9
  __int64 *v24; // rdx
  unsigned int k; // ecx
  __int64 v26; // rdx
  __int64 *v27; // rdx
  unsigned int v28; // r8d
  unsigned int v29; // edx
  __int64 *v30; // rcx
  unsigned int v31; // r8d
  unsigned int v32; // edi
  __int64 v33; // rdx
  __int64 v34; // r15
  _QWORD *v35; // rdx
  __int64 *v36; // rsi
  unsigned int j; // ecx
  __int64 v38; // rdx
  __int64 *v39; // rdx
  __int64 v40; // r14
  unsigned int *v41; // r15
  __int64 v42; // r13
  unsigned int v43; // edx
  unsigned int v44; // r10d
  unsigned int v45; // ecx
  __int64 v46; // r8
  _QWORD *v47; // rax
  unsigned int *v48; // r11
  unsigned int v49; // r9d
  unsigned int v50; // edi
  unsigned int v51; // ecx
  unsigned int v52; // r9d
  _QWORD *v53; // rsi
  __int64 v54; // rcx
  __int64 v55; // r12
  _QWORD *v56; // rbx
  _QWORD *v57; // rcx
  unsigned int m; // esi
  __int64 v59; // rcx
  _QWORD *v60; // rcx
  unsigned int v61; // r12d
  __int64 *v62; // rsi
  unsigned int v63; // edi
  unsigned int v64; // ebx
  __int64 v65; // rcx
  __int64 v66; // rax
  _QWORD *v67; // rcx
  _QWORD *v68; // rcx
  unsigned int v69; // edx
  __int64 v70; // rsi
  _QWORD *v71; // rsi
  signed int v72; // [rsp+8h] [rbp-48h]
  _QWORD *v73; // [rsp+8h] [rbp-48h]
  unsigned int *v77; // [rsp+20h] [rbp-30h]

  v4 = a2 - 1;
  v72 = v4;
  if ( !v4 )
    return;
  v6 = v4;
  v7 = (__int64 **)(a1 + 8LL * v4);
  v8 = &a3[v6];
  for ( i = (unsigned int *)(a4 + v6 * 4); ; --i )
  {
    v10 = *v8;
    v11 = *i;
    --v72;
    if ( *v8 != *i )
    {
      v12 = v72;
      if ( v72 == -1 )
        goto LABEL_18;
      do
      {
        v13 = v11 - v10;
        v14 = *v7;
        v15 = a3[v12];
        v16 = *(_QWORD *)(a1 + 8 * v12);
        if ( v13 <= 0 )
        {
          v29 = 11 - v15;
          v30 = *v7;
          if ( 11 - v15 > v10 )
            v29 = v10;
          v31 = -v13;
          if ( v29 <= v31 )
            v31 = v29;
          v32 = v15 + v31;
          if ( v31 )
          {
            do
            {
              v33 = v15;
              v34 = *v30;
              ++v15;
              v30 += 2;
              v35 = (_QWORD *)(v16 + 16 * v33);
              *v35 = v34;
              v35[1] = *(v30 - 1);
            }
            while ( v32 != v15 );
          }
          v36 = v14;
          for ( j = v31; v10 != j; *(v36 - 1) = v39[1] )
          {
            v38 = j++;
            v36 += 2;
            v39 = &v14[2 * v38];
            *(v36 - 2) = *v39;
          }
          v13 = -v31;
        }
        else
        {
          v17 = v10 - 1;
          if ( v13 > v15 )
            v13 = a3[v12];
          if ( 11 - v10 <= v13 )
            v13 = 11 - v10;
          if ( v10 )
          {
            v18 = v17;
            v19 = v13 + v17;
            v20 = &v14[2 * v18];
            do
            {
              v21 = v19;
              v22 = *v20;
              v23 = v20;
              --v19;
              v20 -= 2;
              v24 = &v14[2 * v21];
              *v24 = v22;
              v24[1] = v20[3];
            }
            while ( v23 != v14 );
          }
          for ( k = v15 - v13; v15 != k; *(v14 - 1) = v27[1] )
          {
            v26 = k++;
            v14 += 2;
            v27 = (__int64 *)(v16 + 16 * v26);
            *(v14 - 2) = *v27;
          }
        }
        a3[v12] -= v13;
        v28 = *v8 + v13;
        *v8 = v28;
        v10 = v28;
        v11 = *i;
        if ( v10 >= *i )
          break;
        --v12;
      }
      while ( (_DWORD)v12 != -1 );
    }
    if ( !v72 )
      break;
LABEL_18:
    --v8;
    --v7;
  }
  if ( a2 > 1 )
  {
    v40 = a1;
    v41 = a3;
    v42 = 1;
    v77 = a3;
    do
    {
      v43 = *v41;
      v44 = v42;
      v45 = *(_DWORD *)(a4 + 4 * v42 - 4);
      if ( *v41 != v45 && a2 != (_DWORD)v42 )
      {
        do
        {
          v46 = *(_QWORD *)(v40 + 8 * v42 - 8);
          v47 = *(_QWORD **)(v40 + 8LL * v44);
          v48 = &v77[v44];
          v49 = *v48;
          if ( (int)(v43 - v45) <= 0 )
          {
            v61 = v45 - v43;
            v62 = *(__int64 **)(v40 + 8LL * v44);
            if ( 11 - v43 <= v45 - v43 )
              v61 = 11 - v43;
            v63 = v61;
            if ( v49 <= v61 )
              v63 = *v48;
            v64 = v63 + v43;
            if ( v63 )
            {
              v73 = *(_QWORD **)(v40 + 8LL * v44);
              do
              {
                v65 = v43;
                v66 = *v62;
                ++v43;
                v62 += 2;
                v67 = (_QWORD *)(v46 + 16 * v65);
                *v67 = v66;
                v67[1] = *(v62 - 1);
              }
              while ( v43 != v64 );
              v47 = v73;
            }
            v68 = v47;
            v69 = v63;
            if ( v49 > v61 )
            {
              do
              {
                v70 = v69++;
                v68 += 2;
                v71 = &v47[2 * v70];
                *(v68 - 2) = *v71;
                *(v68 - 1) = v71[1];
              }
              while ( v49 != v69 );
            }
            v50 = -v63;
          }
          else
          {
            v50 = v43 - v45;
            if ( v43 - v45 > v43 )
              v50 = v43;
            if ( v50 > 11 - v49 )
              v50 = 11 - v49;
            v51 = v49 - 1;
            if ( v49 )
            {
              v52 = v50 + v51;
              v53 = &v47[2 * v51];
              do
              {
                v54 = v52;
                v55 = *v53;
                v56 = v53;
                --v52;
                v53 -= 2;
                v57 = &v47[2 * v54];
                *v57 = v55;
                v57[1] = v53[3];
              }
              while ( v47 != v56 );
            }
            for ( m = v43 - v50; m != v43; *(v47 - 1) = v60[1] )
            {
              v59 = m++;
              v47 += 2;
              v60 = (_QWORD *)(v46 + 16 * v59);
              *(v47 - 2) = *v60;
            }
          }
          *v48 += v50;
          v43 = *v41 - v50;
          *v41 = v43;
          v45 = *(_DWORD *)(a4 + 4 * v42 - 4);
          if ( v43 >= v45 )
            break;
          ++v44;
        }
        while ( a2 != v44 );
      }
      ++v42;
      ++v41;
    }
    while ( a2 != v42 );
  }
}
