// Function: sub_2752980
// Address: 0x2752980
//
__int64 __fastcall sub_2752980(__int64 a1, __int64 a2, __int64 *a3)
{
  int v5; // r13d
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r14
  int v13; // eax
  __int64 v14; // rcx
  _QWORD *v15; // rax
  __int64 *v16; // rsi
  __int64 v17; // rdi
  _QWORD *v18; // rdx
  __int64 v19; // rax
  _QWORD *v20; // r14
  _QWORD *v21; // r13
  __int64 v22; // r9
  unsigned int v23; // eax
  _QWORD *v24; // rcx
  __int64 v25; // rdi
  unsigned int v26; // esi
  int v27; // eax
  int v28; // r10d
  __int64 v29; // r11
  unsigned int v30; // edx
  _QWORD *v31; // r8
  __int64 v32; // rdi
  int v33; // eax
  unsigned int v34; // esi
  __int64 v35; // r9
  __int64 v36; // r8
  unsigned int v37; // ecx
  __int64 *v38; // rdx
  __int64 v39; // rdi
  __int64 *v40; // r10
  int v41; // eax
  __int64 v42; // rax
  int v43; // r11d
  int v44; // eax
  int v45; // eax
  int v46; // r10d
  __int64 v47; // r11
  int v48; // esi
  _QWORD *v49; // rcx
  unsigned int v50; // edx
  __int64 v51; // rdi
  int v52; // r11d
  int v53; // r11d
  __int64 v54; // rcx
  int v55; // edi
  __int64 *v56; // rsi
  int v57; // r10d
  int v58; // r10d
  __int64 *v59; // rcx
  int v60; // esi
  __int64 v61; // r11
  __int64 v62; // rdi
  int v63; // esi
  int v64; // [rsp+8h] [rbp-58h]
  int v65; // [rsp+14h] [rbp-4Ch]
  int v66; // [rsp+14h] [rbp-4Ch]
  unsigned int v67; // [rsp+14h] [rbp-4Ch]
  __int64 v68; // [rsp+18h] [rbp-48h]
  const void *v69; // [rsp+20h] [rbp-40h]

  sub_F54ED0((unsigned __int8 *)a1);
  sub_11C4E30((unsigned __int8 *)a1, 0, 0);
  v69 = (const void *)(a2 + 48);
  v68 = a2 + 32;
  v5 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  if ( v5 )
  {
    v6 = 0;
    while ( 1 )
    {
      if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
        v7 = *(_QWORD *)(a1 - 8) + 32 * v6;
      else
        v7 = a1 + 32 * (v6 - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
      v11 = *(_QWORD *)v7;
      if ( *(_QWORD *)v7 )
      {
        v8 = *(_QWORD *)(v7 + 8);
        **(_QWORD **)(v7 + 16) = v8;
        if ( v8 )
          *(_QWORD *)(v8 + 16) = *(_QWORD *)(v7 + 16);
      }
      *(_QWORD *)v7 = 0;
      if ( *(_QWORD *)(v11 + 16) || a1 == v11 || *(_BYTE *)v11 <= 0x1Cu || !sub_F50EE0((unsigned __int8 *)v11, a3) )
        goto LABEL_11;
      v13 = *(_DWORD *)(a2 + 16);
      if ( v13 )
      {
        v34 = *(_DWORD *)(a2 + 24);
        if ( v34 )
        {
          v35 = v34 - 1;
          v36 = *(_QWORD *)(a2 + 8);
          v37 = v35 & (((unsigned int)v11 >> 4) ^ ((unsigned int)v11 >> 9));
          v38 = (__int64 *)(v36 + 8LL * v37);
          v39 = *v38;
          if ( v11 == *v38 )
            goto LABEL_11;
          v66 = 1;
          v40 = 0;
          v64 = *(_DWORD *)(a2 + 16);
          while ( v39 != -4096 )
          {
            if ( v39 == -8192 && !v40 )
              v40 = v38;
            v37 = v35 & (v66 + v37);
            v38 = (__int64 *)(v36 + 8LL * v37);
            v39 = *v38;
            if ( v11 == *v38 )
              goto LABEL_11;
            ++v66;
          }
          if ( v40 )
            v38 = v40;
          ++*(_QWORD *)a2;
          v41 = v13 + 1;
          if ( 4 * (v64 + 1) < 3 * v34 )
          {
            if ( v34 - *(_DWORD *)(a2 + 20) - v41 <= v34 >> 3 )
            {
              v67 = ((unsigned int)v11 >> 4) ^ ((unsigned int)v11 >> 9);
              sub_CF4090(a2, v34);
              v57 = *(_DWORD *)(a2 + 24);
              if ( !v57 )
              {
LABEL_122:
                ++*(_DWORD *)(a2 + 16);
                BUG();
              }
              v58 = v57 - 1;
              v36 = *(_QWORD *)(a2 + 8);
              v59 = 0;
              v60 = 1;
              LODWORD(v61) = v58 & v67;
              v38 = (__int64 *)(v36 + 8LL * (v58 & v67));
              v62 = *v38;
              v41 = *(_DWORD *)(a2 + 16) + 1;
              if ( v11 != *v38 )
              {
                while ( v62 != -4096 )
                {
                  if ( !v59 && v62 == -8192 )
                    v59 = v38;
                  v35 = (unsigned int)(v60 + 1);
                  v61 = v58 & (unsigned int)(v61 + v60);
                  v38 = (__int64 *)(v36 + 8 * v61);
                  v62 = *v38;
                  if ( v11 == *v38 )
                    goto LABEL_45;
                  ++v60;
                }
                if ( v59 )
                  v38 = v59;
              }
            }
            goto LABEL_45;
          }
        }
        else
        {
          ++*(_QWORD *)a2;
        }
        sub_CF4090(a2, 2 * v34);
        v52 = *(_DWORD *)(a2 + 24);
        if ( !v52 )
          goto LABEL_122;
        v53 = v52 - 1;
        v35 = *(_QWORD *)(a2 + 8);
        LODWORD(v54) = v53 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v38 = (__int64 *)(v35 + 8LL * (unsigned int)v54);
        v36 = *v38;
        v41 = *(_DWORD *)(a2 + 16) + 1;
        if ( v11 != *v38 )
        {
          v55 = 1;
          v56 = 0;
          while ( v36 != -4096 )
          {
            if ( v36 == -8192 && !v56 )
              v56 = v38;
            v54 = v53 & (unsigned int)(v54 + v55);
            v38 = (__int64 *)(v35 + 8 * v54);
            v36 = *v38;
            if ( v11 == *v38 )
              goto LABEL_45;
            ++v55;
          }
          if ( v56 )
            v38 = v56;
        }
LABEL_45:
        *(_DWORD *)(a2 + 16) = v41;
        if ( *v38 != -4096 )
          --*(_DWORD *)(a2 + 20);
        *v38 = v11;
        v42 = *(unsigned int *)(a2 + 40);
        if ( v42 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 44) )
        {
          sub_C8D5F0(v68, v69, v42 + 1, 8u, v36, v35);
          v42 = *(unsigned int *)(a2 + 40);
        }
        *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8 * v42) = v11;
        ++*(_DWORD *)(a2 + 40);
        goto LABEL_11;
      }
      v14 = *(unsigned int *)(a2 + 40);
      v15 = *(_QWORD **)(a2 + 32);
      v16 = &v15[v14];
      v17 = (8 * v14) >> 3;
      if ( (8 * v14) >> 5 )
      {
        v18 = &v15[4 * ((8 * v14) >> 5)];
        while ( *v15 != v11 )
        {
          if ( v15[1] == v11 )
          {
            ++v15;
            goto LABEL_23;
          }
          if ( v15[2] == v11 )
          {
            v15 += 2;
            goto LABEL_23;
          }
          if ( v15[3] == v11 )
          {
            v15 += 3;
            goto LABEL_23;
          }
          v15 += 4;
          if ( v18 == v15 )
          {
            v17 = v16 - v15;
            goto LABEL_51;
          }
        }
        goto LABEL_23;
      }
LABEL_51:
      if ( v17 == 2 )
        goto LABEL_62;
      if ( v17 != 3 )
      {
        if ( v17 != 1 )
          goto LABEL_24;
LABEL_54:
        if ( *v15 != v11 )
          goto LABEL_24;
        goto LABEL_23;
      }
      if ( *v15 != v11 )
        break;
LABEL_23:
      if ( v16 != v15 )
        goto LABEL_11;
LABEL_24:
      if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 44) )
      {
        sub_C8D5F0(v68, v69, v14 + 1, 8u, v9, v10);
        v16 = (__int64 *)(*(_QWORD *)(a2 + 32) + 8LL * *(unsigned int *)(a2 + 40));
      }
      *v16 = v11;
      v19 = (unsigned int)(*(_DWORD *)(a2 + 40) + 1);
      *(_DWORD *)(a2 + 40) = v19;
      if ( (unsigned int)v19 > 0x10 )
      {
        v65 = v5;
        v20 = *(_QWORD **)(a2 + 32);
        v21 = &v20[v19];
        while ( 1 )
        {
          v26 = *(_DWORD *)(a2 + 24);
          if ( !v26 )
            break;
          v22 = *(_QWORD *)(a2 + 8);
          v23 = (v26 - 1) & (((unsigned int)*v20 >> 9) ^ ((unsigned int)*v20 >> 4));
          v24 = (_QWORD *)(v22 + 8LL * v23);
          v25 = *v24;
          if ( *v20 != *v24 )
          {
            v43 = 1;
            v31 = 0;
            while ( v25 != -4096 )
            {
              if ( v25 != -8192 || v31 )
                v24 = v31;
              v23 = (v26 - 1) & (v43 + v23);
              v25 = *(_QWORD *)(v22 + 8LL * v23);
              if ( *v20 == v25 )
                goto LABEL_29;
              ++v43;
              v31 = v24;
              v24 = (_QWORD *)(v22 + 8LL * v23);
            }
            v44 = *(_DWORD *)(a2 + 16);
            if ( !v31 )
              v31 = v24;
            ++*(_QWORD *)a2;
            v33 = v44 + 1;
            if ( 4 * v33 < 3 * v26 )
            {
              if ( v26 - *(_DWORD *)(a2 + 20) - v33 <= v26 >> 3 )
              {
                sub_CF4090(a2, v26);
                v45 = *(_DWORD *)(a2 + 24);
                if ( !v45 )
                {
LABEL_123:
                  ++*(_DWORD *)(a2 + 16);
                  BUG();
                }
                v46 = v45 - 1;
                v47 = *(_QWORD *)(a2 + 8);
                v48 = 1;
                v49 = 0;
                v50 = (v45 - 1) & (((unsigned int)*v20 >> 9) ^ ((unsigned int)*v20 >> 4));
                v31 = (_QWORD *)(v47 + 8LL * v50);
                v51 = *v31;
                v33 = *(_DWORD *)(a2 + 16) + 1;
                if ( *v31 != *v20 )
                {
                  while ( v51 != -4096 )
                  {
                    if ( !v49 && v51 == -8192 )
                      v49 = v31;
                    v50 = v46 & (v48 + v50);
                    v31 = (_QWORD *)(v47 + 8LL * v50);
                    v51 = *v31;
                    if ( *v20 == *v31 )
                      goto LABEL_34;
                    ++v48;
                  }
LABEL_73:
                  if ( v49 )
                    v31 = v49;
                }
              }
LABEL_34:
              *(_DWORD *)(a2 + 16) = v33;
              if ( *v31 != -4096 )
                --*(_DWORD *)(a2 + 20);
              *v31 = *v20;
              goto LABEL_29;
            }
LABEL_32:
            sub_CF4090(a2, 2 * v26);
            v27 = *(_DWORD *)(a2 + 24);
            if ( !v27 )
              goto LABEL_123;
            v28 = v27 - 1;
            v29 = *(_QWORD *)(a2 + 8);
            v30 = (v27 - 1) & (((unsigned int)*v20 >> 9) ^ ((unsigned int)*v20 >> 4));
            v31 = (_QWORD *)(v29 + 8LL * v30);
            v32 = *v31;
            v33 = *(_DWORD *)(a2 + 16) + 1;
            if ( *v20 != *v31 )
            {
              v63 = 1;
              v49 = 0;
              while ( v32 != -4096 )
              {
                if ( !v49 && v32 == -8192 )
                  v49 = v31;
                v30 = v28 & (v63 + v30);
                v31 = (_QWORD *)(v29 + 8LL * v30);
                v32 = *v31;
                if ( *v20 == *v31 )
                  goto LABEL_34;
                ++v63;
              }
              goto LABEL_73;
            }
            goto LABEL_34;
          }
LABEL_29:
          if ( v21 == ++v20 )
          {
            v5 = v65;
            goto LABEL_11;
          }
        }
        ++*(_QWORD *)a2;
        goto LABEL_32;
      }
LABEL_11:
      if ( v5 == (_DWORD)++v6 )
        goto LABEL_14;
    }
    ++v15;
LABEL_62:
    if ( *v15 != v11 )
    {
      ++v15;
      goto LABEL_54;
    }
    goto LABEL_23;
  }
LABEL_14:
  sub_B43D60((_QWORD *)a1);
  return 1;
}
