// Function: sub_9BBB30
// Address: 0x9bbb30
//
unsigned int *__fastcall sub_9BBB30(__int64 a1)
{
  unsigned int *result; // rax
  __int64 v2; // r11
  unsigned int *v3; // rbx
  __int64 v5; // r15
  __int64 v6; // r9
  __int64 v7; // r8
  __int64 *v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rsi
  __int64 *v11; // rdi
  __int64 v12; // rcx
  unsigned int v13; // esi
  __int64 v14; // r13
  __int64 v15; // r14
  int v16; // esi
  int v17; // esi
  __int64 v18; // r9
  __int64 v19; // rcx
  __int64 *v20; // rax
  __int64 v21; // rdi
  int v22; // edx
  int v23; // r11d
  __int64 *v24; // r8
  int v25; // ecx
  __int64 v26; // rcx
  unsigned int *v27; // rdx
  int v28; // esi
  int v29; // esi
  __int64 v30; // r9
  int v31; // r11d
  __int64 v32; // rcx
  __int64 v33; // rdi
  int v34; // [rsp-4Ch] [rbp-4Ch]
  unsigned int v35; // [rsp-4Ch] [rbp-4Ch]
  __int64 v36; // [rsp-48h] [rbp-48h]
  unsigned int *v37; // [rsp-40h] [rbp-40h]

  result = *(unsigned int **)(a1 + 32);
  if ( result )
  {
    v2 = *((_QWORD *)result + 2);
    if ( *(_BYTE *)(v2 + 232) )
    {
      v3 = *(unsigned int **)(v2 + 240);
      result = &v3[3 * *(unsigned int *)(v2 + 248)];
      v37 = result;
      if ( result != v3 )
      {
        v5 = v2;
        v36 = a1 + 144;
        while ( 1 )
        {
          v12 = *(_QWORD *)(v5 + 56);
          v13 = *(_DWORD *)(a1 + 168);
          v14 = v3[1];
          v15 = *(_QWORD *)(v12 + 8LL * *v3);
          if ( v13 )
          {
            v6 = *(_QWORD *)(a1 + 152);
            v7 = (v13 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
            v8 = (__int64 *)(v6 + 56 * v7);
            v9 = *v8;
            if ( v15 == *v8 )
            {
LABEL_6:
              v10 = *(_QWORD *)(v12 + 8 * v14);
              v11 = v8 + 1;
              if ( !*((_BYTE *)v8 + 36) )
                goto LABEL_7;
              goto LABEL_30;
            }
            v34 = 1;
            v20 = 0;
            while ( v9 != -4096 )
            {
              if ( v9 == -8192 && !v20 )
                v20 = v8;
              LODWORD(v7) = (v13 - 1) & (v34 + v7);
              v8 = (__int64 *)(v6 + 56LL * (unsigned int)v7);
              v9 = *v8;
              if ( v15 == *v8 )
                goto LABEL_6;
              ++v34;
            }
            v25 = *(_DWORD *)(a1 + 160);
            if ( !v20 )
              v20 = v8;
            ++*(_QWORD *)(a1 + 144);
            v22 = v25 + 1;
            if ( 4 * (v25 + 1) < 3 * v13 )
            {
              if ( v13 - *(_DWORD *)(a1 + 164) - v22 > v13 >> 3 )
                goto LABEL_27;
              v35 = ((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4);
              sub_9BB8E0(v36, v13);
              v28 = *(_DWORD *)(a1 + 168);
              if ( !v28 )
              {
LABEL_52:
                ++*(_DWORD *)(a1 + 160);
                BUG();
              }
              v29 = v28 - 1;
              v30 = *(_QWORD *)(a1 + 152);
              v24 = 0;
              v31 = 1;
              LODWORD(v32) = v29 & v35;
              v20 = (__int64 *)(v30 + 56LL * (v29 & v35));
              v33 = *v20;
              v22 = *(_DWORD *)(a1 + 160) + 1;
              if ( v15 == *v20 )
                goto LABEL_27;
              while ( v33 != -4096 )
              {
                if ( v33 == -8192 && !v24 )
                  v24 = v20;
                v32 = v29 & (unsigned int)(v32 + v31);
                v20 = (__int64 *)(v30 + 56 * v32);
                v33 = *v20;
                if ( v15 == *v20 )
                  goto LABEL_27;
                ++v31;
              }
              goto LABEL_39;
            }
          }
          else
          {
            ++*(_QWORD *)(a1 + 144);
          }
          sub_9BB8E0(v36, 2 * v13);
          v16 = *(_DWORD *)(a1 + 168);
          if ( !v16 )
            goto LABEL_52;
          v17 = v16 - 1;
          v18 = *(_QWORD *)(a1 + 152);
          LODWORD(v19) = v17 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
          v20 = (__int64 *)(v18 + 56LL * (unsigned int)v19);
          v21 = *v20;
          v22 = *(_DWORD *)(a1 + 160) + 1;
          if ( v15 == *v20 )
            goto LABEL_27;
          v23 = 1;
          v24 = 0;
          while ( v21 != -4096 )
          {
            if ( !v24 && v21 == -8192 )
              v24 = v20;
            v19 = v17 & (unsigned int)(v19 + v23);
            v20 = (__int64 *)(v18 + 56 * v19);
            v21 = *v20;
            if ( v15 == *v20 )
              goto LABEL_27;
            ++v23;
          }
LABEL_39:
          if ( v24 )
            v20 = v24;
LABEL_27:
          *(_DWORD *)(a1 + 160) = v22;
          if ( *v20 != -4096 )
            --*(_DWORD *)(a1 + 164);
          *v20 = v15;
          v11 = v20 + 1;
          v20[1] = 0;
          v20[2] = (__int64)(v20 + 5);
          v20[3] = 2;
          *((_DWORD *)v20 + 8) = 0;
          *((_BYTE *)v20 + 36) = 1;
          v10 = *(_QWORD *)(*(_QWORD *)(v5 + 56) + 8 * v14);
LABEL_30:
          result = (unsigned int *)v11[1];
          v26 = *((unsigned int *)v11 + 5);
          v27 = &result[2 * v26];
          if ( result != v27 )
          {
            while ( v10 != *(_QWORD *)result )
            {
              result += 2;
              if ( v27 == result )
                goto LABEL_33;
            }
            goto LABEL_8;
          }
LABEL_33:
          if ( (unsigned int)v26 >= *((_DWORD *)v11 + 4) )
          {
LABEL_7:
            result = (unsigned int *)sub_C8CC70(v11, v10);
LABEL_8:
            v3 += 3;
            if ( v37 == v3 )
              return result;
          }
          else
          {
            v3 += 3;
            *((_DWORD *)v11 + 5) = v26 + 1;
            *(_QWORD *)v27 = v10;
            ++*v11;
            if ( v37 == v3 )
              return result;
          }
        }
      }
    }
  }
  return result;
}
