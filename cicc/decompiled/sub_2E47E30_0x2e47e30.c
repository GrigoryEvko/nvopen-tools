// Function: sub_2E47E30
// Address: 0x2e47e30
//
unsigned __int64 __fastcall sub_2E47E30(__int64 a1, unsigned int a2, __int64 a3, int a4)
{
  __int64 v6; // rcx
  __int16 *v7; // r14
  unsigned __int64 result; // rax
  int v9; // r12d
  __int64 v10; // rsi
  unsigned int v11; // ecx
  _QWORD *v12; // rdx
  int v13; // edi
  __int64 v14; // r13
  unsigned int v15; // esi
  int v16; // r11d
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rdi
  int v24; // edx
  int v25; // edi
  int v26; // ecx
  _QWORD *v27; // rdi
  __int64 v28; // rsi
  int v29; // r8d
  const void *v30; // r9
  int v31; // edx
  int v32; // r9d
  __int64 v33; // rsi
  int v34; // edx
  __int64 *v35; // rcx
  __int64 v36; // rdi
  __int64 v37; // rax
  _QWORD *v38; // rdi
  int v39; // ecx
  int v40; // r8d
  int v41; // esi
  int v42; // esi
  __int64 v43; // rdx
  __int64 v44; // rdi
  int v45; // r11d
  int v46; // esi
  int v47; // esi
  int v48; // r11d
  __int64 v49; // rdx
  __int64 v50; // rdi
  __int64 v51; // [rsp+0h] [rbp-50h]
  unsigned int v52; // [rsp+8h] [rbp-48h]
  __int64 v54[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
  v7 = (__int16 *)(*(_QWORD *)(*(_QWORD *)a1 + 56LL) + 2LL * (*(_DWORD *)(v6 + 24LL * a2 + 16) >> 12));
  result = a1 + 144;
  v9 = *(_DWORD *)(v6 + 24LL * a2 + 16) & 0xFFF;
  v51 = a1 + 144;
  while ( v7 )
  {
    result = *(unsigned int *)(a1 + 200);
    v10 = *(_QWORD *)(a1 + 184);
    if ( (_DWORD)result )
    {
      v11 = (result - 1) & (37 * v9);
      v12 = (_QWORD *)(v10 + ((unsigned __int64)v11 << 7));
      v13 = *(_DWORD *)v12;
      if ( v9 != *(_DWORD *)v12 )
      {
        v31 = 1;
        while ( v13 != -1 )
        {
          v32 = v31 + 1;
          v11 = (result - 1) & (v31 + v11);
          v12 = (_QWORD *)(v10 + ((unsigned __int64)v11 << 7));
          v13 = *(_DWORD *)v12;
          if ( v9 == *(_DWORD *)v12 )
            goto LABEL_5;
          v31 = v32;
        }
        goto LABEL_12;
      }
LABEL_5:
      result = v10 + (result << 7);
      if ( v12 != (_QWORD *)result )
      {
        v14 = v12[1];
        v54[0] = v14;
        if ( v14 )
        {
          if ( a4 != 1 )
          {
            v15 = *(_DWORD *)(a1 + 168);
            if ( v15 )
            {
              v16 = 1;
              v17 = *(_QWORD *)(a1 + 152);
              v18 = ((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4);
              v19 = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
              v20 = v17 + 56 * v19;
              v21 = 0;
              v22 = *(_QWORD *)v20;
              if ( v14 == *(_QWORD *)v20 )
              {
LABEL_10:
                v23 = v20 + 8;
                if ( !*(_BYTE *)(v20 + 36) )
                {
LABEL_11:
                  result = (unsigned __int64)sub_C8CC70(v23, a3, v18, v20, v19, v17);
                  goto LABEL_12;
                }
LABEL_27:
                result = *(_QWORD *)(v23 + 8);
                v20 = *(unsigned int *)(v23 + 20);
                v18 = result + 8 * v20;
                if ( result != v18 )
                {
                  while ( a3 != *(_QWORD *)result )
                  {
                    result += 8LL;
                    if ( v18 == result )
                      goto LABEL_30;
                  }
                  goto LABEL_12;
                }
LABEL_30:
                if ( (unsigned int)v20 < *(_DWORD *)(v23 + 16) )
                {
                  *(_DWORD *)(v23 + 20) = v20 + 1;
                  *(_QWORD *)v18 = a3;
                  ++*(_QWORD *)v23;
                  goto LABEL_12;
                }
                goto LABEL_11;
              }
              while ( v22 != -4096 )
              {
                if ( !v21 && v22 == -8192 )
                  v21 = v20;
                v19 = (v15 - 1) & (v16 + (_DWORD)v19);
                v20 = v17 + 56LL * (unsigned int)v19;
                v22 = *(_QWORD *)v20;
                if ( v14 == *(_QWORD *)v20 )
                  goto LABEL_10;
                ++v16;
              }
              v25 = *(_DWORD *)(a1 + 160);
              if ( !v21 )
                v21 = v20;
              ++*(_QWORD *)(a1 + 144);
              v26 = v25 + 1;
              if ( 4 * (v25 + 1) < 3 * v15 )
              {
                v19 = v15 >> 3;
                if ( v15 - *(_DWORD *)(a1 + 164) - v26 <= (unsigned int)v19 )
                {
                  v52 = ((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4);
                  sub_2E47BE0(v51, v15);
                  v46 = *(_DWORD *)(a1 + 168);
                  if ( !v46 )
                  {
LABEL_70:
                    ++*(_DWORD *)(a1 + 160);
                    BUG();
                  }
                  v47 = v46 - 1;
                  v17 = *(_QWORD *)(a1 + 152);
                  v19 = 0;
                  v48 = 1;
                  LODWORD(v49) = v47 & v52;
                  v21 = v17 + 56LL * (v47 & v52);
                  v50 = *(_QWORD *)v21;
                  v26 = *(_DWORD *)(a1 + 160) + 1;
                  if ( *(_QWORD *)v21 != v14 )
                  {
                    while ( v50 != -4096 )
                    {
                      if ( !v19 && v50 == -8192 )
                        v19 = v21;
                      v49 = v47 & (unsigned int)(v49 + v48);
                      v21 = v17 + 56 * v49;
                      v50 = *(_QWORD *)v21;
                      if ( v14 == *(_QWORD *)v21 )
                        goto LABEL_24;
                      ++v48;
                    }
                    goto LABEL_62;
                  }
                }
                goto LABEL_24;
              }
            }
            else
            {
              ++*(_QWORD *)(a1 + 144);
            }
            sub_2E47BE0(v51, 2 * v15);
            v41 = *(_DWORD *)(a1 + 168);
            if ( !v41 )
              goto LABEL_70;
            v42 = v41 - 1;
            v17 = *(_QWORD *)(a1 + 152);
            LODWORD(v43) = v42 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
            v21 = v17 + 56LL * (unsigned int)v43;
            v44 = *(_QWORD *)v21;
            v26 = *(_DWORD *)(a1 + 160) + 1;
            if ( v14 != *(_QWORD *)v21 )
            {
              v45 = 1;
              v19 = 0;
              while ( v44 != -4096 )
              {
                if ( !v19 && v44 == -8192 )
                  v19 = v21;
                v43 = v42 & (unsigned int)(v43 + v45);
                v21 = v17 + 56 * v43;
                v44 = *(_QWORD *)v21;
                if ( v14 == *(_QWORD *)v21 )
                  goto LABEL_24;
                ++v45;
              }
LABEL_62:
              if ( v19 )
                v21 = v19;
            }
LABEL_24:
            *(_DWORD *)(a1 + 160) = v26;
            if ( *(_QWORD *)v21 != -4096 )
              --*(_DWORD *)(a1 + 164);
            *(_QWORD *)v21 = v14;
            v23 = v21 + 8;
            *(_QWORD *)(v21 + 8) = 0;
            *(_QWORD *)(v21 + 16) = v21 + 40;
            *(_QWORD *)(v21 + 24) = 2;
            *(_DWORD *)(v21 + 32) = 0;
            *(_BYTE *)(v21 + 36) = 1;
            goto LABEL_27;
          }
          if ( *(_DWORD *)(a1 + 48) )
          {
            result = *(unsigned int *)(a1 + 56);
            v33 = *(_QWORD *)(a1 + 40);
            if ( !(_DWORD)result )
              goto LABEL_12;
            v34 = result - 1;
            result = ((_DWORD)result - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
            v35 = (__int64 *)(v33 + 8 * result);
            v36 = *v35;
            if ( v14 != *v35 )
            {
              v39 = 1;
              while ( v36 != -4096 )
              {
                v40 = v39 + 1;
                result = v34 & (unsigned int)(v39 + result);
                v35 = (__int64 *)(v33 + 8LL * (unsigned int)result);
                v36 = *v35;
                if ( v14 == *v35 )
                  goto LABEL_43;
                v39 = v40;
              }
              goto LABEL_12;
            }
LABEL_43:
            *v35 = -8192;
            v37 = *(unsigned int *)(a1 + 72);
            --*(_DWORD *)(a1 + 48);
            v38 = *(_QWORD **)(a1 + 64);
            ++*(_DWORD *)(a1 + 52);
            v28 = (__int64)&v38[v37];
            result = (unsigned __int64)sub_2E44940(v38, v28, v54);
            v30 = (const void *)(result + 8);
            if ( result + 8 != v28 )
            {
LABEL_35:
              result = (unsigned __int64)memmove((void *)result, v30, v28 - (_QWORD)v30);
              v29 = *(_DWORD *)(a1 + 72);
            }
LABEL_36:
            *(_DWORD *)(a1 + 72) = v29 - 1;
            goto LABEL_12;
          }
          v27 = *(_QWORD **)(a1 + 64);
          v28 = (__int64)&v27[*(unsigned int *)(a1 + 72)];
          result = (unsigned __int64)sub_2E44940(v27, v28, v54);
          if ( v28 != result )
          {
            v30 = (const void *)(result + 8);
            if ( v28 != result + 8 )
              goto LABEL_35;
            goto LABEL_36;
          }
        }
      }
    }
LABEL_12:
    v24 = *v7++;
    v9 += v24;
    if ( !(_WORD)v24 )
      return result;
  }
  return result;
}
