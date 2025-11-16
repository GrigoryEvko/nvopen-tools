// Function: sub_34A7CD0
// Address: 0x34a7cd0
//
void __fastcall sub_34A7CD0(__int64 a1, unsigned int a2, unsigned int *a3, __int64 a4)
{
  signed int v4; // esi
  __int64 v6; // r12
  _QWORD *v7; // r13
  unsigned int *v8; // rbx
  unsigned int *i; // r12
  unsigned int v10; // r8d
  unsigned int v11; // r9d
  __int64 v12; // r11
  int v13; // r9d
  _QWORD *v14; // rax
  unsigned int v15; // ecx
  __int64 v16; // rsi
  unsigned int v17; // edi
  __int64 v18; // rdx
  unsigned int v19; // edi
  _QWORD *v20; // rdx
  __int64 v21; // r15
  __int64 v22; // r8
  _QWORD *v23; // r14
  unsigned int k; // edx
  __int64 v25; // rdi
  unsigned int v26; // r9d
  unsigned int v27; // edx
  unsigned int v28; // r9d
  _QWORD *v29; // rdx
  unsigned int v30; // r14d
  __int64 v31; // r15
  __int64 v32; // rdi
  _QWORD *v33; // rcx
  unsigned int j; // edx
  __int64 v35; // rsi
  __int64 v36; // r14
  unsigned int *v37; // r15
  __int64 v38; // r13
  unsigned int v39; // edx
  unsigned int v40; // r10d
  unsigned int v41; // ecx
  int v42; // r12d
  _QWORD *v43; // rax
  __int64 v44; // rcx
  unsigned int *v45; // r11
  unsigned int v46; // r9d
  unsigned int v47; // edi
  unsigned int v48; // r8d
  __int64 v49; // rsi
  unsigned int v50; // r8d
  _QWORD *v51; // rsi
  __int64 v52; // r12
  __int64 v53; // r9
  _QWORD *v54; // rbx
  unsigned int m; // esi
  __int64 v56; // r8
  unsigned int v57; // r12d
  _QWORD *v58; // rsi
  unsigned int v59; // edi
  unsigned int v60; // ebx
  __int64 v61; // rax
  __int64 v62; // r8
  _QWORD *v63; // rcx
  unsigned int v64; // edx
  __int64 v65; // rsi
  signed int v66; // [rsp+8h] [rbp-48h]
  _QWORD *v67; // [rsp+8h] [rbp-48h]
  unsigned int *v71; // [rsp+20h] [rbp-30h]

  v4 = a2 - 1;
  v66 = v4;
  if ( !v4 )
    return;
  v6 = v4;
  v7 = (_QWORD *)(a1 + 8LL * v4);
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
        goto LABEL_18;
      do
      {
        v13 = v11 - v10;
        v14 = (_QWORD *)*v7;
        v15 = a3[v12];
        v16 = *(_QWORD *)(a1 + 8 * v12);
        if ( v13 <= 0 )
        {
          v27 = 12 - v15;
          if ( 12 - v15 > v10 )
            v27 = v10;
          v28 = -v13;
          if ( v27 <= v28 )
            v28 = v27;
          v29 = v14 + 12;
          v30 = v15 + v28;
          if ( v28 )
          {
            do
            {
              v31 = *(v29 - 12);
              v32 = v15++;
              ++v29;
              *(_QWORD *)(v16 + 8 * v32) = v31;
              *(_QWORD *)(v16 + 8 * v32 + 96) = *(v29 - 1);
            }
            while ( v15 != v30 );
          }
          v33 = v14;
          for ( j = v28; v10 != j; v33[11] = v14[v35 + 12] )
          {
            v35 = j++;
            *v33++ = v14[v35];
          }
          v13 = -v28;
        }
        else
        {
          v17 = v10 - 1;
          if ( v13 > v15 )
            v13 = a3[v12];
          if ( 12 - v10 <= v13 )
            v13 = 12 - v10;
          if ( v10 )
          {
            v18 = v17;
            v19 = v13 + v17;
            v20 = &v14[v18];
            do
            {
              v21 = *v20;
              v22 = v19;
              v23 = v20;
              --v19;
              --v20;
              v14[v22] = v21;
              v14[v22 + 12] = v20[13];
            }
            while ( v23 != v14 );
          }
          for ( k = v15 - v13; v15 != k; v14[11] = *(_QWORD *)(v16 + 8 * v25 + 96) )
          {
            v25 = k++;
            *v14++ = *(_QWORD *)(v16 + 8 * v25);
          }
        }
        a3[v12] -= v13;
        v26 = *v8 + v13;
        *v8 = v26;
        v10 = v26;
        v11 = *i;
        if ( v10 >= *i )
          break;
        --v12;
      }
      while ( (_DWORD)v12 != -1 );
    }
    if ( !v66 )
      break;
LABEL_18:
    --v8;
    --v7;
  }
  if ( a2 > 1 )
  {
    v36 = a1;
    v37 = a3;
    v38 = 1;
    v71 = a3;
    do
    {
      v39 = *v37;
      v40 = v38;
      v41 = *(_DWORD *)(a4 + 4 * v38 - 4);
      if ( *v37 != v41 && a2 != (_DWORD)v38 )
      {
        do
        {
          v42 = v39 - v41;
          v43 = *(_QWORD **)(v36 + 8LL * v40);
          v44 = *(_QWORD *)(v36 + 8 * v38 - 8);
          v45 = &v71[v40];
          v46 = *v45;
          if ( v42 <= 0 )
          {
            v57 = -v42;
            if ( 12 - v39 <= v57 )
              v57 = 12 - v39;
            v58 = v43 + 12;
            v59 = v57;
            if ( v46 <= v57 )
              v59 = *v45;
            v60 = v59 + v39;
            if ( v59 )
            {
              v67 = *(_QWORD **)(v36 + 8LL * v40);
              do
              {
                v61 = *(v58 - 12);
                v62 = v39++;
                ++v58;
                *(_QWORD *)(v44 + 8 * v62) = v61;
                *(_QWORD *)(v44 + 8 * v62 + 96) = *(v58 - 1);
              }
              while ( v39 != v60 );
              v43 = v67;
            }
            v63 = v43;
            v64 = v59;
            if ( v46 > v57 )
            {
              do
              {
                v65 = v64++;
                *v63++ = v43[v65];
                v63[11] = v43[v65 + 12];
              }
              while ( v46 != v64 );
            }
            v47 = -v59;
          }
          else
          {
            v47 = v42;
            v48 = v46 - 1;
            if ( v42 > v39 )
              v47 = v39;
            if ( v47 > 12 - v46 )
              v47 = 12 - v46;
            if ( v46 )
            {
              v49 = v48;
              v50 = v47 + v48;
              v51 = &v43[v49];
              do
              {
                v52 = *v51;
                v53 = v50;
                v54 = v51;
                --v50;
                --v51;
                v43[v53] = v52;
                v43[v53 + 12] = v51[13];
              }
              while ( v43 != v54 );
            }
            for ( m = v39 - v47; m != v39; v43[11] = *(_QWORD *)(v44 + 8 * v56 + 96) )
            {
              v56 = m++;
              *v43++ = *(_QWORD *)(v44 + 8 * v56);
            }
          }
          *v45 += v47;
          v39 = *v37 - v47;
          *v37 = v39;
          v41 = *(_DWORD *)(a4 + 4 * v38 - 4);
          if ( v39 >= v41 )
            break;
          ++v40;
        }
        while ( a2 != v40 );
      }
      ++v38;
      ++v37;
    }
    while ( a2 != v38 );
  }
}
