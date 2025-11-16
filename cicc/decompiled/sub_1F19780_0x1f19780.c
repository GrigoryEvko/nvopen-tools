// Function: sub_1F19780
// Address: 0x1f19780
//
void __fastcall sub_1F19780(__int64 a1, unsigned int a2, unsigned int *a3, __int64 a4)
{
  signed int v4; // esi
  __int64 v6; // r15
  unsigned int *v7; // r14
  unsigned int *i; // r15
  unsigned int v9; // r8d
  unsigned int v10; // r10d
  __int64 v11; // r12
  int v12; // r10d
  unsigned int v13; // esi
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rdx
  unsigned int v17; // r8d
  __int64 v18; // r11
  __int64 v19; // rdx
  __int64 v20; // r9
  _QWORD *v21; // rcx
  __int64 v22; // rsi
  unsigned int v23; // r8d
  __int64 v24; // rdx
  __int64 v25; // r9
  _QWORD *v26; // rcx
  unsigned int v27; // r10d
  unsigned int v28; // edx
  unsigned int v29; // r10d
  __int64 v30; // rdx
  unsigned int v31; // r11d
  __int64 v32; // r9
  _QWORD *v33; // rcx
  unsigned int v34; // esi
  __int64 j; // rdx
  __int64 v36; // rdi
  _QWORD *v37; // rcx
  unsigned int *v38; // r15
  __int64 v39; // r14
  unsigned int v40; // ecx
  unsigned int v41; // r12d
  unsigned int v42; // edx
  unsigned int v43; // ebx
  __int64 v44; // rax
  __int64 v45; // rdi
  unsigned int *v46; // r13
  unsigned int v47; // r9d
  unsigned int v48; // edx
  int v49; // r10d
  __int64 v50; // rdx
  unsigned int v51; // r8d
  __int64 v52; // rdx
  __int64 v53; // r11
  __int64 v54; // r9
  _QWORD *v55; // rsi
  __int64 v56; // rcx
  unsigned int v57; // r8d
  __int64 v58; // rdx
  __int64 v59; // r9
  _QWORD *v60; // rsi
  unsigned int v61; // ebx
  unsigned int v62; // r10d
  __int64 v63; // rdx
  unsigned int v64; // r11d
  __int64 v65; // r8
  _QWORD *v66; // rsi
  unsigned int v67; // esi
  __int64 v68; // rdx
  __int64 v69; // rdi
  _QWORD *v70; // rcx
  signed int v71; // [rsp+0h] [rbp-58h]
  unsigned int v74; // [rsp+18h] [rbp-40h]
  unsigned int *v75; // [rsp+18h] [rbp-40h]
  __int64 *v76; // [rsp+20h] [rbp-38h]
  unsigned int v77; // [rsp+20h] [rbp-38h]

  v4 = a2 - 1;
  v71 = v4;
  if ( !v4 )
    return;
  v6 = v4;
  v76 = (__int64 *)(a1 + 8LL * v4);
  v7 = &a3[v6];
  for ( i = (unsigned int *)(a4 + v6 * 4); ; --i )
  {
    v9 = *v7;
    v10 = *i;
    --v71;
    if ( *v7 != *i )
    {
      v11 = v71;
      if ( v71 == -1 )
        goto LABEL_20;
      do
      {
        v12 = v10 - v9;
        v13 = a3[v11];
        v14 = *v76;
        v15 = *(_QWORD *)(a1 + 8 * v11);
        if ( v12 <= 0 )
        {
          v28 = 9 - v13;
          if ( 9 - v13 > v9 )
            v28 = v9;
          v29 = -v12;
          if ( v28 <= v29 )
            v29 = v28;
          v30 = 0;
          v31 = v13 + v29;
          if ( v29 )
          {
            do
            {
              v32 = v13++;
              v33 = (_QWORD *)(v15 + 16 * v32);
              *v33 = *(_QWORD *)(v14 + 4 * v30);
              v33[1] = *(_QWORD *)(v14 + 4 * v30 + 8);
              LODWORD(v33) = *(_DWORD *)(v14 + v30 + 144);
              v30 += 4;
              *(_DWORD *)(v15 + 4 * v32 + 144) = (_DWORD)v33;
            }
            while ( v13 != v31 );
          }
          v34 = v29;
          for ( j = 0; v9 != v34; j += 4 )
          {
            v36 = v34++;
            v37 = (_QWORD *)(v14 + 16 * v36);
            *(_QWORD *)(v14 + 4 * j) = *v37;
            *(_QWORD *)(v14 + 4 * j + 8) = v37[1];
            *(_DWORD *)(v14 + j + 144) = *(_DWORD *)(v14 + 4 * v36 + 144);
          }
          v12 = -v29;
        }
        else
        {
          if ( v12 > v13 )
            v12 = a3[v11];
          if ( 9 - v9 <= v12 )
            v12 = 9 - v9;
          v16 = v9 - 1;
          if ( v9 )
          {
            v74 = a3[v11];
            v17 = v12 + v16;
            v18 = -3 * v14;
            v19 = v14 + 4 * v16 + 144;
            do
            {
              v20 = v17--;
              v21 = (_QWORD *)(v14 + 16 * v20);
              *v21 = *(_QWORD *)(v18 + 4 * v19 - 576);
              v22 = *(_QWORD *)(v18 + 4 * v19 - 568);
              v19 -= 4;
              v21[1] = v22;
              *(_DWORD *)(v14 + 4 * v20 + 144) = *(_DWORD *)(v19 + 4);
            }
            while ( v14 + 140 != v19 );
            v13 = v74;
          }
          v23 = v13 - v12;
          if ( v13 != v13 - v12 )
          {
            v24 = 0;
            do
            {
              v25 = v23++;
              v26 = (_QWORD *)(v15 + 16 * v25);
              *(_QWORD *)(v14 + 4 * v24) = *v26;
              *(_QWORD *)(v14 + 4 * v24 + 8) = v26[1];
              *(_DWORD *)(v14 + v24 + 144) = *(_DWORD *)(v15 + 4 * v25 + 144);
              v24 += 4;
            }
            while ( v13 != v23 );
          }
        }
        a3[v11] -= v12;
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
    if ( !v71 )
      break;
LABEL_20:
    --v76;
    --v7;
  }
  if ( a2 > 1 )
  {
    v38 = a3;
    v39 = 1;
    v75 = a3;
    do
    {
      v40 = *v38;
      v41 = v39;
      v42 = *(_DWORD *)(a4 + 4 * v39 - 4);
      if ( *v38 != v42 && a2 != (_DWORD)v39 )
      {
        do
        {
          v43 = v40 - v42;
          v44 = *(_QWORD *)(a1 + 8LL * v41);
          v45 = *(_QWORD *)(a1 + 8 * v39 - 8);
          v46 = &v75[v41];
          v47 = *v46;
          if ( (int)(v40 - v42) <= 0 )
          {
            v61 = v42 - v40;
            if ( 9 - v40 <= v42 - v40 )
              v61 = 9 - v40;
            v62 = v61;
            if ( v47 <= v61 )
              v62 = *v46;
            v63 = 0;
            v64 = v62 + v40;
            if ( v62 )
            {
              do
              {
                v65 = v40++;
                v66 = (_QWORD *)(v45 + 16 * v65);
                *v66 = *(_QWORD *)(v44 + 4 * v63);
                v66[1] = *(_QWORD *)(v44 + 4 * v63 + 8);
                LODWORD(v66) = *(_DWORD *)(v44 + v63 + 144);
                v63 += 4;
                *(_DWORD *)(v45 + 4 * v65 + 144) = (_DWORD)v66;
              }
              while ( v40 != v64 );
            }
            v67 = v62;
            v68 = 0;
            if ( v47 > v61 )
            {
              do
              {
                v69 = v67++;
                v70 = (_QWORD *)(v44 + 16 * v69);
                *(_QWORD *)(v44 + 4 * v68) = *v70;
                *(_QWORD *)(v44 + 4 * v68 + 8) = v70[1];
                *(_DWORD *)(v44 + v68 + 144) = *(_DWORD *)(v44 + 4 * v69 + 144);
                v68 += 4;
              }
              while ( v47 != v67 );
            }
            v49 = -v62;
          }
          else
          {
            if ( v43 > v40 )
              v43 = v40;
            v48 = 9 - v47;
            if ( v43 <= 9 - v47 )
              v48 = v43;
            v49 = v48;
            v50 = v47 - 1;
            if ( v47 )
            {
              v77 = v40;
              v51 = v49 + v50;
              v52 = v44 + 4 * v50 + 144;
              v53 = -3 * v44;
              do
              {
                v54 = v51--;
                v55 = (_QWORD *)(v44 + 16 * v54);
                *v55 = *(_QWORD *)(v53 + 4 * v52 - 576);
                v56 = *(_QWORD *)(v53 + 4 * v52 - 568);
                v52 -= 4;
                v55[1] = v56;
                *(_DWORD *)(v44 + 4 * v54 + 144) = *(_DWORD *)(v52 + 4);
              }
              while ( v52 != v44 + 140 );
              v40 = v77;
            }
            v57 = v40 - v49;
            if ( v40 - v49 != v40 )
            {
              v58 = 0;
              do
              {
                v59 = v57++;
                v60 = (_QWORD *)(v45 + 16 * v59);
                *(_QWORD *)(v44 + 4 * v58) = *v60;
                *(_QWORD *)(v44 + 4 * v58 + 8) = v60[1];
                *(_DWORD *)(v44 + v58 + 144) = *(_DWORD *)(v45 + 4 * v59 + 144);
                v58 += 4;
              }
              while ( v57 != v40 );
            }
          }
          *v46 += v49;
          v40 = *v38 - v49;
          *v38 = v40;
          v42 = *(_DWORD *)(a4 + 4 * v39 - 4);
          if ( v40 >= v42 )
            break;
          ++v41;
        }
        while ( a2 != v41 );
      }
      ++v39;
      ++v38;
    }
    while ( a2 != v39 );
  }
}
