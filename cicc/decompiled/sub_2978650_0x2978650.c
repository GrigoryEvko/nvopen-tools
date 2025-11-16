// Function: sub_2978650
// Address: 0x2978650
//
_QWORD **__fastcall sub_2978650(__int64 a1)
{
  __int64 v1; // r11
  _QWORD **result; // rax
  __int64 v4; // r14
  __int64 v5; // r13
  unsigned int v6; // r8d
  _QWORD *v7; // r11
  unsigned int v8; // r10d
  __int64 v9; // r9
  int v10; // r12d
  _QWORD *v11; // rdx
  unsigned int v12; // ebx
  unsigned int v13; // esi
  _QWORD *v14; // rax
  _QWORD *v15; // rcx
  unsigned __int64 v16; // rbx
  __int64 v17; // rdi
  unsigned int v18; // r12d
  unsigned int v19; // esi
  unsigned __int64 *v20; // rax
  unsigned __int64 v21; // rcx
  unsigned __int64 v22; // rbx
  int v23; // eax
  int v24; // eax
  int v25; // eax
  unsigned __int64 *v26; // rdx
  int v27; // eax
  int v28; // eax
  int v29; // esi
  int v30; // esi
  __int64 v31; // r8
  unsigned int v32; // ecx
  unsigned __int64 v33; // rdi
  int v34; // r12d
  unsigned __int64 *v35; // r10
  int v36; // ecx
  int v37; // ecx
  __int64 v38; // rdi
  unsigned __int64 *v39; // r9
  unsigned int v40; // r12d
  int v41; // r10d
  unsigned __int64 v42; // rsi
  int v43; // esi
  int v44; // esi
  __int64 v45; // r8
  __int64 v46; // rcx
  __int64 v47; // rdi
  int v48; // ebx
  _QWORD *v49; // r9
  int v50; // ecx
  int v51; // ecx
  __int64 v52; // rdi
  _QWORD *v53; // r8
  __int64 v54; // rbx
  int v55; // r10d
  __int64 v56; // rsi
  int v57; // [rsp+8h] [rbp-48h]
  _QWORD *v58; // [rsp+8h] [rbp-48h]
  _QWORD *v59; // [rsp+8h] [rbp-48h]
  _QWORD *v60; // [rsp+8h] [rbp-48h]
  _QWORD *v61; // [rsp+8h] [rbp-48h]
  _QWORD **v62; // [rsp+10h] [rbp-40h]
  _QWORD **v63; // [rsp+18h] [rbp-38h]

  result = **(_QWORD ****)a1;
  v63 = result;
  v62 = &result[*(unsigned int *)(*(_QWORD *)a1 + 8LL)];
  if ( v62 != result )
  {
    v4 = v1;
    do
    {
      v5 = *(_QWORD *)(a1 + 8);
      v6 = *(_DWORD *)(v5 + 24);
      v7 = *v63;
      if ( v6 )
      {
        v8 = v6 - 1;
        v9 = *(_QWORD *)(v5 + 8);
        v10 = 1;
        v11 = 0;
        v12 = ((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4);
        v13 = (v6 - 1) & v12;
        v14 = (_QWORD *)(v9 + 16LL * v13);
        v15 = (_QWORD *)*v14;
        if ( v7 == (_QWORD *)*v14 )
        {
LABEL_5:
          v16 = v14[1];
          v17 = *(_QWORD *)(a1 + 16);
          if ( v16 )
          {
            while ( 1 )
            {
              if ( *(_QWORD *)(v16 + 40) == v17 )
                goto LABEL_13;
              v18 = ((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4);
              v19 = v18 & v8;
              v20 = (unsigned __int64 *)(v9 + 16LL * (v18 & v8));
              v21 = *v20;
              if ( *v20 != v16 )
                break;
LABEL_8:
              v16 = v20[1];
              if ( !v16 )
                goto LABEL_9;
            }
            v57 = 1;
            v26 = 0;
            while ( v21 != -4096 )
            {
              if ( !v26 && v21 == -8192 )
                v26 = v20;
              v19 = v8 & (v57 + v19);
              v20 = (unsigned __int64 *)(v9 + 16LL * v19);
              v21 = *v20;
              if ( *v20 == v16 )
                goto LABEL_8;
              ++v57;
            }
            if ( !v26 )
              v26 = v20;
            v27 = *(_DWORD *)(v5 + 16);
            ++*(_QWORD *)v5;
            v28 = v27 + 1;
            if ( 4 * v28 >= 3 * v6 )
            {
              v58 = v7;
              sub_2978470(v5, 2 * v6);
              v29 = *(_DWORD *)(v5 + 24);
              if ( !v29 )
                goto LABEL_91;
              v30 = v29 - 1;
              v31 = *(_QWORD *)(v5 + 8);
              v7 = v58;
              v32 = v30 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
              v28 = *(_DWORD *)(v5 + 16) + 1;
              v26 = (unsigned __int64 *)(v31 + 16LL * v32);
              v33 = *v26;
              if ( v16 != *v26 )
              {
                v34 = 1;
                v35 = 0;
                while ( v33 != -4096 )
                {
                  if ( v33 == -8192 && !v35 )
                    v35 = v26;
                  v32 = v30 & (v34 + v32);
                  v26 = (unsigned __int64 *)(v31 + 16LL * v32);
                  v33 = *v26;
                  if ( v16 == *v26 )
                    goto LABEL_34;
                  ++v34;
                }
                if ( v35 )
                  v26 = v35;
              }
            }
            else if ( v6 - *(_DWORD *)(v5 + 20) - v28 <= v6 >> 3 )
            {
              v59 = v7;
              sub_2978470(v5, v6);
              v36 = *(_DWORD *)(v5 + 24);
              if ( !v36 )
              {
LABEL_91:
                ++*(_DWORD *)(v5 + 16);
                BUG();
              }
              v37 = v36 - 1;
              v38 = *(_QWORD *)(v5 + 8);
              v39 = 0;
              v40 = v37 & v18;
              v7 = v59;
              v41 = 1;
              v28 = *(_DWORD *)(v5 + 16) + 1;
              v26 = (unsigned __int64 *)(v38 + 16LL * v40);
              v42 = *v26;
              if ( *v26 != v16 )
              {
                while ( v42 != -4096 )
                {
                  if ( !v39 && v42 == -8192 )
                    v39 = v26;
                  v40 = v37 & (v41 + v40);
                  v26 = (unsigned __int64 *)(v38 + 16LL * v40);
                  v42 = *v26;
                  if ( v16 == *v26 )
                    goto LABEL_34;
                  ++v41;
                }
                if ( v39 )
                  v26 = v39;
              }
            }
LABEL_34:
            *(_DWORD *)(v5 + 16) = v28;
            if ( *v26 != -4096 )
              --*(_DWORD *)(v5 + 20);
            *v26 = v16;
            v26[1] = 0;
            v17 = *(_QWORD *)(a1 + 16);
          }
          goto LABEL_9;
        }
        while ( v15 != (_QWORD *)-4096LL )
        {
          if ( !v11 && v15 == (_QWORD *)-8192LL )
            v11 = v14;
          v13 = v8 & (v10 + v13);
          v14 = (_QWORD *)(v9 + 16LL * v13);
          v15 = (_QWORD *)*v14;
          if ( v7 == (_QWORD *)*v14 )
            goto LABEL_5;
          ++v10;
        }
        if ( !v11 )
          v11 = v14;
        v24 = *(_DWORD *)(v5 + 16);
        ++*(_QWORD *)v5;
        v25 = v24 + 1;
        if ( 4 * v25 < 3 * v6 )
        {
          if ( v6 - *(_DWORD *)(v5 + 20) - v25 <= v6 >> 3 )
          {
            v61 = v7;
            sub_2978470(v5, v6);
            v50 = *(_DWORD *)(v5 + 24);
            if ( !v50 )
            {
LABEL_93:
              ++*(_DWORD *)(v5 + 16);
              BUG();
            }
            v51 = v50 - 1;
            v52 = *(_QWORD *)(v5 + 8);
            v53 = 0;
            LODWORD(v54) = v51 & v12;
            v7 = v61;
            v55 = 1;
            v25 = *(_DWORD *)(v5 + 16) + 1;
            v11 = (_QWORD *)(v52 + 16LL * (unsigned int)v54);
            v56 = *v11;
            if ( v61 != (_QWORD *)*v11 )
            {
              while ( v56 != -4096 )
              {
                if ( !v53 && v56 == -8192 )
                  v53 = v11;
                v54 = v51 & (unsigned int)(v54 + v55);
                v11 = (_QWORD *)(v52 + 16 * v54);
                v56 = *v11;
                if ( v61 == (_QWORD *)*v11 )
                  goto LABEL_25;
                ++v55;
              }
              if ( v53 )
                v11 = v53;
            }
          }
          goto LABEL_25;
        }
      }
      else
      {
        ++*(_QWORD *)v5;
      }
      v60 = v7;
      sub_2978470(v5, 2 * v6);
      v43 = *(_DWORD *)(v5 + 24);
      if ( !v43 )
        goto LABEL_93;
      v7 = v60;
      v44 = v43 - 1;
      v45 = *(_QWORD *)(v5 + 8);
      LODWORD(v46) = v44 & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
      v25 = *(_DWORD *)(v5 + 16) + 1;
      v11 = (_QWORD *)(v45 + 16LL * (unsigned int)v46);
      v47 = *v11;
      if ( v60 != (_QWORD *)*v11 )
      {
        v48 = 1;
        v49 = 0;
        while ( v47 != -4096 )
        {
          if ( !v49 && v47 == -8192 )
            v49 = v11;
          v46 = v44 & (unsigned int)(v46 + v48);
          v11 = (_QWORD *)(v45 + 16 * v46);
          v47 = *v11;
          if ( v60 == (_QWORD *)*v11 )
            goto LABEL_25;
          ++v48;
        }
        if ( v49 )
          v11 = v49;
      }
LABEL_25:
      *(_DWORD *)(v5 + 16) = v25;
      if ( *v11 != -4096 )
        --*(_DWORD *)(v5 + 20);
      *v11 = v7;
      v11[1] = 0;
      v17 = *(_QWORD *)(a1 + 16);
LABEL_9:
      v22 = *(_QWORD *)(v17 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v22 == v17 + 48 )
      {
        v16 = 0;
      }
      else
      {
        if ( !v22 )
          BUG();
        v23 = *(unsigned __int8 *)(v22 - 24);
        v16 = v22 - 24;
        if ( (unsigned int)(v23 - 30) >= 0xB )
          v16 = 0;
      }
LABEL_13:
      LOWORD(v4) = 0;
      sub_B444E0(v7, v16 + 24, v4);
      result = ++v63;
    }
    while ( v62 != v63 );
  }
  return result;
}
