// Function: sub_30FDFD0
// Address: 0x30fdfd0
//
void __fastcall sub_30FDFD0(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  unsigned int v7; // edi
  char v8; // r8
  unsigned __int64 *v9; // rsi
  __int64 v10; // rax
  unsigned __int64 v11; // rbx
  unsigned __int64 *v12; // rdx
  unsigned __int64 *v13; // rax
  unsigned __int64 *v14; // rcx
  unsigned __int64 *v15; // rdx
  _QWORD *v16; // rax
  __int64 v17; // rdi
  _QWORD *v18; // rax
  _QWORD *v19; // rsi
  __int64 v20; // rdx
  _QWORD *v21; // rax
  _QWORD *v22; // r14
  __int64 v23; // rbx
  _QWORD *v24; // r12
  unsigned int v25; // esi
  unsigned __int64 v26; // rbx
  __int64 v27; // rdi
  int v28; // r10d
  __int64 v29; // rax
  _QWORD *v30; // rax
  __int64 *v31; // r12
  __int64 *v32; // r14
  char v33; // di
  __int64 v34; // rsi
  _QWORD *v35; // rax
  int v36; // eax
  int v37; // eax
  __int64 v38; // rdx
  __int64 v39; // rcx
  _QWORD *v40; // rax
  _QWORD *v41; // rax
  unsigned __int64 v42; // r15
  __int64 v43; // rax
  _QWORD *v44; // rax
  _QWORD *v45; // rdx
  char v46; // di
  __int64 *v47; // rax
  int v48; // ecx
  __int64 v49; // rdi
  __int64 v50; // rsi
  int v51; // r11d
  int v52; // edx
  __int64 v53; // rsi
  __int64 v54; // rdi
  int v55; // r10d
  unsigned int v56; // r15d
  __int64 v58; // [rsp+10h] [rbp-60h]
  __int64 v59; // [rsp+20h] [rbp-50h]
  __int64 v60; // [rsp+28h] [rbp-48h]
  _QWORD *v61; // [rsp+28h] [rbp-48h]
  int v62; // [rsp+34h] [rbp-3Ch]
  _QWORD *v63; // [rsp+38h] [rbp-38h]

  if ( !a2 || *(_BYTE *)(a1 + 360) )
    return;
  sub_30FAE30(*(_QWORD *)(a1 + 136));
  *(_QWORD *)(a1 + 136) = 0;
  v7 = *(_DWORD *)(a1 + 276);
  *(_QWORD *)(a1 + 144) = a1 + 128;
  *(_QWORD *)(a1 + 152) = a1 + 128;
  v59 = a1 + 256;
  v58 = a1 + 296;
  *(_QWORD *)(a1 + 160) = 0;
  if ( v7 != *(_DWORD *)(a1 + 280) )
  {
    v63 = (_QWORD *)(a1 + 208);
    while ( 1 )
    {
      v8 = *(_BYTE *)(a1 + 284);
      v9 = *(unsigned __int64 **)(a1 + 264);
      if ( v8 )
      {
        v10 = v7;
        v11 = *v9;
        v12 = &v9[v7];
        if ( v9 == v12 )
          goto LABEL_14;
      }
      else
      {
        v11 = *v9;
        v12 = &v9[*(unsigned int *)(a1 + 272)];
        if ( v9 == v12 )
          goto LABEL_84;
      }
      v13 = *(unsigned __int64 **)(a1 + 264);
      while ( 1 )
      {
        v11 = *v13;
        v14 = v13;
        if ( *v13 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v12 == ++v13 )
        {
          v11 = v14[1];
          break;
        }
      }
      if ( v8 )
      {
        v10 = v7;
LABEL_14:
        v15 = &v9[v10];
        v16 = *(_QWORD **)(a1 + 264);
        if ( v9 != v15 )
        {
          while ( *v16 != v11 )
          {
            if ( v15 == ++v16 )
              goto LABEL_19;
          }
          v17 = v7 - 1;
          *(_DWORD *)(a1 + 276) = v17;
          *v16 = v9[v17];
          ++*(_QWORD *)(a1 + 256);
        }
        goto LABEL_19;
      }
LABEL_84:
      v47 = sub_C8CA60(v59, v11);
      if ( v47 )
      {
        *v47 = -2;
        ++*(_DWORD *)(a1 + 280);
        ++*(_QWORD *)(a1 + 256);
      }
LABEL_19:
      *(_QWORD *)(a1 + 184) += sub_30FCC90(a1, *(_QWORD *)(v11 + 8));
      v18 = *(_QWORD **)(a1 + 216);
      if ( !v18 )
        goto LABEL_116;
      v19 = (_QWORD *)(a1 + 208);
      do
      {
        while ( 1 )
        {
          v4 = v18[2];
          v20 = v18[3];
          if ( v18[4] >= v11 )
            break;
          v18 = (_QWORD *)v18[3];
          if ( !v20 )
            goto LABEL_24;
        }
        v19 = v18;
        v18 = (_QWORD *)v18[2];
      }
      while ( v4 );
LABEL_24:
      if ( v19 == v63 || v19[4] > v11 )
LABEL_116:
        sub_426320((__int64)"map::at");
      v3 = *(unsigned int *)(v11 + 32);
      v62 = *((_DWORD *)v19 + 10);
      v21 = *(_QWORD **)(v11 + 24);
      v22 = &v21[v3];
      if ( v21 == v22 )
      {
LABEL_37:
        v7 = *(_DWORD *)(a1 + 276);
        if ( v7 == *(_DWORD *)(a1 + 280) )
          break;
      }
      else
      {
        do
        {
          v23 = *v21;
          v24 = v21;
          v3 = *v21 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v3 && *(_QWORD *)v3 )
          {
            while ( 1 )
            {
              if ( v22 == v24 )
                goto LABEL_37;
              v25 = *(_DWORD *)(a1 + 320);
              v26 = v23 & 0xFFFFFFFFFFFFFFF8LL;
              if ( v25 )
              {
                v5 = v25 - 1;
                v27 = *(_QWORD *)(a1 + 304);
                v6 = 0;
                v28 = 1;
                v4 = (unsigned int)v5 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
                v3 = v27 + 8 * v4;
                v29 = *(_QWORD *)v3;
                if ( v26 == *(_QWORD *)v3 )
                  goto LABEL_34;
                while ( v29 != -4096 )
                {
                  if ( v6 || v29 != -8192 )
                    v3 = v6;
                  v6 = (unsigned int)(v28 + 1);
                  v4 = (unsigned int)v5 & (v28 + (_DWORD)v4);
                  v29 = *(_QWORD *)(v27 + 8LL * (unsigned int)v4);
                  if ( v26 == v29 )
                    goto LABEL_34;
                  ++v28;
                  v6 = v3;
                  v3 = v27 + 8LL * (unsigned int)v4;
                }
                v36 = *(_DWORD *)(a1 + 312);
                if ( !v6 )
                  v6 = v3;
                ++*(_QWORD *)(a1 + 296);
                v37 = v36 + 1;
                if ( 4 * v37 < 3 * v25 )
                {
                  v38 = v25 - *(_DWORD *)(a1 + 316) - v37;
                  v39 = v25 >> 3;
                  if ( (unsigned int)v38 <= (unsigned int)v39 )
                  {
                    sub_30FDE00(v58, v25);
                    v52 = *(_DWORD *)(a1 + 320);
                    if ( !v52 )
                    {
LABEL_117:
                      ++*(_DWORD *)(a1 + 312);
                      BUG();
                    }
                    v38 = (unsigned int)(v52 - 1);
                    v53 = *(_QWORD *)(a1 + 304);
                    v54 = 0;
                    v55 = 1;
                    v56 = v38 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
                    v6 = v53 + 8LL * v56;
                    v39 = *(_QWORD *)v6;
                    v37 = *(_DWORD *)(a1 + 312) + 1;
                    if ( v26 != *(_QWORD *)v6 )
                    {
                      while ( v39 != -4096 )
                      {
                        if ( v39 == -8192 && !v54 )
                          v54 = v6;
                        v5 = (unsigned int)(v55 + 1);
                        v56 = v38 & (v55 + v56);
                        v6 = v53 + 8LL * v56;
                        v39 = *(_QWORD *)v6;
                        if ( v26 == *(_QWORD *)v6 )
                          goto LABEL_57;
                        ++v55;
                      }
                      if ( v54 )
                        v6 = v54;
                    }
                  }
                  goto LABEL_57;
                }
              }
              else
              {
                ++*(_QWORD *)(a1 + 296);
              }
              sub_30FDE00(v58, 2 * v25);
              v48 = *(_DWORD *)(a1 + 320);
              if ( !v48 )
                goto LABEL_117;
              v39 = (unsigned int)(v48 - 1);
              v49 = *(_QWORD *)(a1 + 304);
              v38 = (unsigned int)v39 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
              v6 = v49 + 8 * v38;
              v50 = *(_QWORD *)v6;
              v37 = *(_DWORD *)(a1 + 312) + 1;
              if ( v26 != *(_QWORD *)v6 )
              {
                v5 = 0;
                v51 = 1;
                while ( v50 != -4096 )
                {
                  if ( !v5 && v50 == -8192 )
                    v5 = v6;
                  v38 = (unsigned int)v39 & (v51 + (_DWORD)v38);
                  v6 = v49 + 8LL * (unsigned int)v38;
                  v50 = *(_QWORD *)v6;
                  if ( v26 == *(_QWORD *)v6 )
                    goto LABEL_57;
                  ++v51;
                }
                if ( v5 )
                  v6 = v5;
              }
LABEL_57:
              *(_DWORD *)(a1 + 312) = v37;
              if ( *(_QWORD *)v6 != -4096 )
                --*(_DWORD *)(a1 + 316);
              *(_QWORD *)v6 = v26;
              ++*(_QWORD *)(a1 + 176);
              if ( !*(_BYTE *)(a1 + 284) )
                goto LABEL_80;
              v40 = *(_QWORD **)(a1 + 264);
              v39 = *(unsigned int *)(a1 + 276);
              v38 = (__int64)&v40[v39];
              if ( v40 != (_QWORD *)v38 )
              {
                while ( v26 != *v40 )
                {
                  if ( (_QWORD *)v38 == ++v40 )
                    goto LABEL_81;
                }
                goto LABEL_64;
              }
LABEL_81:
              if ( (unsigned int)v39 < *(_DWORD *)(a1 + 272) )
              {
                *(_DWORD *)(a1 + 276) = v39 + 1;
                *(_QWORD *)v38 = v26;
                ++*(_QWORD *)(a1 + 256);
              }
              else
              {
LABEL_80:
                sub_C8CC70(v59, v26, v38, v39, v5, v6);
              }
LABEL_64:
              v41 = *(_QWORD **)(a1 + 216);
              v42 = a1 + 208;
              if ( !v41 )
                goto LABEL_71;
              do
              {
                while ( 1 )
                {
                  v4 = v41[2];
                  v3 = v41[3];
                  if ( v41[4] >= v26 )
                    break;
                  v41 = (_QWORD *)v41[3];
                  if ( !v3 )
                    goto LABEL_69;
                }
                v42 = (unsigned __int64)v41;
                v41 = (_QWORD *)v41[2];
              }
              while ( v4 );
LABEL_69:
              if ( (_QWORD *)v42 == v63 || *(_QWORD *)(v42 + 32) > v26 )
              {
LABEL_71:
                v60 = v42;
                v43 = sub_22077B0(0x30u);
                *(_QWORD *)(v43 + 32) = v26;
                v42 = v43;
                *(_DWORD *)(v43 + 40) = 0;
                v44 = sub_30FDD00((_QWORD *)(a1 + 200), v60, (unsigned __int64 *)(v43 + 32));
                if ( v45 )
                {
                  v46 = v63 == v45 || v44 || v45[4] > v26;
                  sub_220F040(v46, v42, v45, v63);
                  ++*(_QWORD *)(a1 + 240);
                }
                else
                {
                  v61 = v44;
                  j_j___libc_free_0(v42);
                  v42 = (unsigned __int64)v61;
                }
              }
              *(_DWORD *)(v42 + 40) = v62;
LABEL_34:
              v30 = v24 + 1;
              if ( v22 == v24 + 1 )
                goto LABEL_37;
              while ( 1 )
              {
                v23 = *v30;
                v24 = v30;
                v3 = *v30 & 0xFFFFFFFFFFFFFFF8LL;
                if ( v3 )
                {
                  if ( *(_QWORD *)v3 )
                    break;
                }
                if ( v22 == ++v30 )
                  goto LABEL_37;
              }
            }
          }
          ++v21;
        }
        while ( v22 != v21 );
        v7 = *(_DWORD *)(a1 + 276);
        if ( v7 == *(_DWORD *)(a1 + 280) )
          break;
      }
    }
  }
  *(_QWORD *)(a1 + 184) -= *(_QWORD *)(a1 + 192);
  *(_QWORD *)(a1 + 192) = 0;
  v31 = *(__int64 **)(a2 + 8);
  v32 = &v31[*(unsigned int *)(a2 + 16)];
  if ( v31 != v32 )
  {
    v33 = *(_BYTE *)(a1 + 284);
    v34 = *v31;
    if ( !v33 )
      goto LABEL_46;
LABEL_40:
    v35 = *(_QWORD **)(a1 + 264);
    v4 = *(unsigned int *)(a1 + 276);
    v3 = (unsigned __int64)&v35[v4];
    if ( v35 == (_QWORD *)v3 )
    {
LABEL_87:
      if ( (unsigned int)v4 >= *(_DWORD *)(a1 + 272) )
      {
LABEL_46:
        while ( 1 )
        {
          ++v31;
          sub_C8CC70(a1 + 256, v34, v3, v4, v5, v6);
          v33 = *(_BYTE *)(a1 + 284);
          if ( v32 == v31 )
            break;
LABEL_45:
          v34 = *v31;
          if ( v33 )
            goto LABEL_40;
        }
      }
      else
      {
        v4 = (unsigned int)(v4 + 1);
        ++v31;
        *(_DWORD *)(a1 + 276) = v4;
        *(_QWORD *)v3 = v34;
        v33 = *(_BYTE *)(a1 + 284);
        ++*(_QWORD *)(a1 + 256);
        if ( v32 != v31 )
          goto LABEL_45;
      }
    }
    else
    {
      while ( v34 != *v35 )
      {
        if ( (_QWORD *)v3 == ++v35 )
          goto LABEL_87;
      }
      if ( v32 != ++v31 )
        goto LABEL_45;
    }
  }
}
