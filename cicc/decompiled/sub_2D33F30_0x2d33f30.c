// Function: sub_2D33F30
// Address: 0x2d33f30
//
void __fastcall sub_2D33F30(__int64 a1, unsigned int a2, unsigned int *a3, __int64 a4)
{
  signed int v4; // esi
  __int64 v6; // r14
  __int64 *v7; // r15
  unsigned int *v8; // r13
  unsigned int *i; // r14
  unsigned int v10; // esi
  unsigned int v11; // r9d
  __int64 v12; // r11
  int v13; // r9d
  __int64 v14; // rax
  unsigned int v15; // r8d
  __int64 v16; // rcx
  __int64 v17; // rdx
  unsigned int v18; // esi
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rdi
  unsigned int v22; // edx
  __int64 v23; // rsi
  __int64 v24; // rdi
  unsigned int v25; // edx
  unsigned int v26; // r9d
  __int64 v27; // rdx
  __int64 v28; // rdi
  int v29; // r12d
  unsigned int v30; // ecx
  __int64 j; // rdx
  __int64 v32; // rdi
  __int64 v33; // rbx
  unsigned int *v34; // r13
  __int64 v35; // r12
  unsigned int v36; // ecx
  unsigned int v37; // r10d
  unsigned int v38; // edx
  __int64 v39; // rsi
  unsigned int v40; // r15d
  __int64 v41; // rax
  unsigned int *v42; // r11
  unsigned int v43; // r8d
  unsigned int v44; // edx
  int v45; // r9d
  __int64 v46; // rdx
  unsigned int v47; // edi
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  unsigned int v51; // edx
  __int64 v52; // rdi
  __int64 v53; // r8
  unsigned int v54; // r15d
  unsigned int v55; // r9d
  __int64 v56; // rdx
  __int64 v57; // rdi
  int v58; // r8d
  unsigned int v59; // ecx
  __int64 v60; // rdx
  __int64 v61; // rsi
  signed int v62; // [rsp+0h] [rbp-50h]
  unsigned int *v66; // [rsp+18h] [rbp-38h]
  __int64 v67; // [rsp+20h] [rbp-30h]
  unsigned int v68; // [rsp+20h] [rbp-30h]
  unsigned int v69; // [rsp+20h] [rbp-30h]

  v4 = a2 - 1;
  v62 = v4;
  if ( !v4 )
    return;
  v6 = v4;
  v7 = (__int64 *)(a1 + 8LL * v4);
  v8 = &a3[v6];
  for ( i = (unsigned int *)(a4 + v6 * 4); ; --i )
  {
    v10 = *v8;
    v11 = *i;
    --v62;
    if ( *v8 != *i )
    {
      v12 = v62;
      if ( v62 == -1 )
        goto LABEL_20;
      do
      {
        v13 = v11 - v10;
        v14 = *v7;
        v15 = a3[v12];
        v16 = *(_QWORD *)(a1 + 8 * v12);
        if ( v13 <= 0 )
        {
          v25 = 16 - v15;
          if ( 16 - v15 > v10 )
            v25 = v10;
          v26 = -v13;
          if ( v25 <= v26 )
            v26 = v25;
          v27 = 0;
          if ( v26 )
          {
            do
            {
              v28 = v15 + (unsigned int)v27;
              *(_QWORD *)(v16 + 8 * v28) = *(_QWORD *)(v14 + 8 * v27);
              v29 = *(_DWORD *)(v14 + 4 * v27++ + 128);
              *(_DWORD *)(v16 + 4 * v28 + 128) = v29;
            }
            while ( v27 != v26 );
          }
          v30 = v26;
          for ( j = 0; v10 != v30; j += 4 )
          {
            v32 = v30++;
            *(_QWORD *)(v14 + 2 * j) = *(_QWORD *)(v14 + 8 * v32);
            *(_DWORD *)(v14 + j + 128) = *(_DWORD *)(v14 + 4 * v32 + 128);
          }
          v13 = -v26;
        }
        else
        {
          if ( v13 > v15 )
            v13 = a3[v12];
          if ( 16 - v10 <= v13 )
            v13 = 16 - v10;
          v17 = v10 - 1;
          if ( v10 )
          {
            v67 = *(_QWORD *)(a1 + 8 * v12);
            v18 = v13 + v17;
            v19 = v14 + 4 * v17 + 128;
            do
            {
              v20 = *(_QWORD *)(2 * v19 - v14 - 256);
              v21 = v18;
              v19 -= 4;
              --v18;
              *(_QWORD *)(v14 + 8 * v21) = v20;
              *(_DWORD *)(v14 + 4 * v21 + 128) = *(_DWORD *)(v19 + 4);
            }
            while ( v14 + 124 != v19 );
            v16 = v67;
          }
          v22 = v15 - v13;
          if ( v15 != v15 - v13 )
          {
            v23 = 0;
            do
            {
              v24 = v22++;
              *(_QWORD *)(v14 + 2 * v23) = *(_QWORD *)(v16 + 8 * v24);
              *(_DWORD *)(v14 + v23 + 128) = *(_DWORD *)(v16 + 4 * v24 + 128);
              v23 += 4;
            }
            while ( v15 != v22 );
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
    if ( !v62 )
      break;
LABEL_20:
    --v8;
    --v7;
  }
  if ( a2 > 1 )
  {
    v33 = a1;
    v34 = a3;
    v35 = 1;
    v66 = a3;
    do
    {
      v36 = *v34;
      v37 = v35;
      v38 = *(_DWORD *)(a4 + 4 * v35 - 4);
      if ( *v34 != v38 && a2 != (_DWORD)v35 )
      {
        do
        {
          v39 = *(_QWORD *)(v33 + 8 * v35 - 8);
          v40 = v36 - v38;
          v41 = *(_QWORD *)(v33 + 8LL * v37);
          v42 = &v66[v37];
          v43 = *v42;
          if ( (int)(v36 - v38) <= 0 )
          {
            v54 = v38 - v36;
            if ( 16 - v36 <= v38 - v36 )
              v54 = 16 - v36;
            v55 = v54;
            if ( v43 <= v54 )
              v55 = *v42;
            v56 = 0;
            if ( v55 )
            {
              v69 = *v42;
              do
              {
                v57 = v36 + (unsigned int)v56;
                *(_QWORD *)(v39 + 8 * v57) = *(_QWORD *)(v41 + 8 * v56);
                v58 = *(_DWORD *)(v41 + 4 * v56++ + 128);
                *(_DWORD *)(v39 + 4 * v57 + 128) = v58;
              }
              while ( v55 != v56 );
              v43 = v69;
            }
            v59 = v55;
            v60 = 0;
            if ( v43 > v54 )
            {
              do
              {
                v61 = v59++;
                *(_QWORD *)(v41 + 2 * v60) = *(_QWORD *)(v41 + 8 * v61);
                *(_DWORD *)(v41 + v60 + 128) = *(_DWORD *)(v41 + 4 * v61 + 128);
                v60 += 4;
              }
              while ( v43 != v59 );
            }
            v45 = -v55;
          }
          else
          {
            if ( v40 > v36 )
              v40 = v36;
            v44 = 16 - v43;
            if ( v40 <= 16 - v43 )
              v44 = v40;
            v45 = v44;
            v46 = v43 - 1;
            if ( v43 )
            {
              v68 = v36;
              v47 = v45 + v46;
              v48 = v41 + 4 * v46 + 128;
              do
              {
                v49 = *(_QWORD *)(2 * v48 - v41 - 256);
                v50 = v47;
                v48 -= 4;
                --v47;
                *(_QWORD *)(v41 + 8 * v50) = v49;
                *(_DWORD *)(v41 + 4 * v50 + 128) = *(_DWORD *)(v48 + 4);
              }
              while ( v48 != v41 + 124 );
              v36 = v68;
            }
            v51 = v36 - v45;
            if ( v36 - v45 != v36 )
            {
              v52 = 0;
              do
              {
                v53 = v51++;
                *(_QWORD *)(v41 + 2 * v52) = *(_QWORD *)(v39 + 8 * v53);
                *(_DWORD *)(v41 + v52 + 128) = *(_DWORD *)(v39 + 4 * v53 + 128);
                v52 += 4;
              }
              while ( v51 != v36 );
            }
          }
          *v42 += v45;
          v36 = *v34 - v45;
          *v34 = v36;
          v38 = *(_DWORD *)(a4 + 4 * v35 - 4);
          if ( v36 >= v38 )
            break;
          ++v37;
        }
        while ( a2 != v37 );
      }
      ++v35;
      ++v34;
    }
    while ( a2 != v35 );
  }
}
