// Function: sub_A20890
// Address: 0xa20890
//
void __fastcall sub_A20890(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 *v3; // r15
  __int64 *v4; // r14
  __int64 v5; // rdx
  __int64 v6; // rax
  _QWORD *v7; // r13
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // r12
  __int64 v11; // r15
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rsi
  __int64 v22; // rax
  unsigned int v23; // edx
  __int64 *v24; // rcx
  __int64 v25; // r11
  __int64 v26; // r13
  __int64 *v27; // r10
  _QWORD *v28; // r12
  _QWORD *v29; // r11
  __int64 v30; // rbx
  __int64 v31; // rax
  int v32; // r12d
  unsigned __int64 v33; // rdx
  _QWORD *v34; // rbx
  __int64 v35; // rax
  int v36; // ecx
  int v37; // ebx
  __int64 v39; // [rsp+10h] [rbp-50h]
  __int64 *v40; // [rsp+10h] [rbp-50h]
  _QWORD *i; // [rsp+28h] [rbp-38h]
  __int64 *v44; // [rsp+28h] [rbp-38h]
  _QWORD *v45; // [rsp+28h] [rbp-38h]
  __int64 *v46; // [rsp+28h] [rbp-38h]

  if ( !*(_DWORD *)(a2 + 16) )
    return;
  v3 = *(__int64 **)(*(_QWORD *)a1 + 8LL);
  v4 = &v3[*(unsigned int *)(*(_QWORD *)a1 + 24LL)];
  if ( *(_DWORD *)(*(_QWORD *)a1 + 16LL) && v4 != v3 )
  {
    while ( (unsigned __int64)*v3 > 0xFFFFFFFFFFFFFFFDLL )
    {
      if ( ++v3 == v4 )
        goto LABEL_4;
    }
LABEL_18:
    if ( v4 != v3 )
    {
      v20 = *v3;
      v21 = *(_QWORD *)(a2 + 8);
      v22 = *(unsigned int *)(a2 + 24);
      if ( (_DWORD)v22 )
      {
        v23 = (v22 - 1) & (((0xBF58476D1CE4E5B9LL * v20) >> 31) ^ (484763065 * v20));
        v24 = (__int64 *)(v21 + 56LL * v23);
        v25 = *v24;
        if ( v20 == *v24 )
        {
LABEL_21:
          v26 = *(_QWORD *)(a1 + 8);
          v39 = *(unsigned int *)(v26 + 8);
          if ( v24 != (__int64 *)(v21 + 56 * v22) )
          {
            v27 = (__int64 *)v24[4];
            v28 = v24 + 2;
            if ( v24 + 2 != v27 )
            {
              v29 = (_QWORD *)v24[4];
              v30 = 0;
              do
              {
                v44 = v27;
                ++v30;
                v31 = sub_220EF30(v29);
                v27 = v44;
                v29 = (_QWORD *)v31;
              }
              while ( v28 != (_QWORD *)v31 );
              v32 = v30;
              v33 = v30 + v39;
              if ( v30 + v39 <= (unsigned __int64)*(unsigned int *)(v26 + 12) )
              {
                v34 = (_QWORD *)(*(_QWORD *)v26 + 16 * v39);
                do
                {
LABEL_27:
                  if ( v34 )
                  {
                    *v34 = v27[4];
                    v34[1] = v27[5];
                  }
                  v45 = v29;
                  v34 += 2;
                  v35 = sub_220EF30(v27);
                  v29 = v45;
                  v27 = (__int64 *)v35;
                }
                while ( v45 != (_QWORD *)v35 );
LABEL_30:
                *(_DWORD *)(v26 + 8) += v32;
                while ( 1 )
                {
                  if ( ++v3 == v4 )
                    goto LABEL_4;
                  if ( (unsigned __int64)*v3 <= 0xFFFFFFFFFFFFFFFDLL )
                    goto LABEL_18;
                }
              }
LABEL_39:
              v40 = v29;
              v46 = v27;
              sub_C8D5F0(v26, v26 + 16, v33, 16);
              v27 = v46;
              v29 = v40;
              v34 = (_QWORD *)(*(_QWORD *)v26 + 16LL * *(unsigned int *)(v26 + 8));
              if ( v40 == v46 )
                goto LABEL_30;
              goto LABEL_27;
            }
            v33 = *(unsigned int *)(v26 + 8);
            v29 = v24 + 2;
LABEL_38:
            v27 = v29;
            v32 = 0;
            if ( *(unsigned int *)(v26 + 12) >= v33 )
              goto LABEL_30;
            goto LABEL_39;
          }
LABEL_37:
          v33 = *(unsigned int *)(v26 + 8);
          v29 = 0;
          goto LABEL_38;
        }
        v36 = 1;
        while ( v25 != -1 )
        {
          v37 = v36 + 1;
          v23 = (v22 - 1) & (v36 + v23);
          v24 = (__int64 *)(v21 + 56LL * v23);
          v25 = *v24;
          if ( v20 == *v24 )
            goto LABEL_21;
          v36 = v37;
        }
      }
      v26 = *(_QWORD *)(a1 + 8);
      goto LABEL_37;
    }
  }
LABEL_4:
  v5 = *(_QWORD *)(a1 + 8);
  v6 = *(unsigned int *)(v5 + 8);
  if ( (_DWORD)v6 )
  {
    v7 = *(_QWORD **)v5;
    v8 = 2 * v6;
    if ( v6 != 1 )
    {
      qsort(*(void **)v5, (v8 * 8) >> 4, 0x10u, (__compar_fn_t)sub_A16990);
      v19 = *(_QWORD *)(a1 + 8);
      v7 = *(_QWORD **)v19;
      v8 = 2LL * *(unsigned int *)(v19 + 8);
    }
    for ( i = &v7[v8]; i != v7; ++*(_DWORD *)(v17 + 8) )
    {
      v9 = v7[1];
      v10 = *v7;
      v11 = *(_QWORD *)(a1 + 16);
      v12 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL);
      v13 = sub_C94890(*v7, v9);
      v14 = sub_C0CA60(v12, v10, (v13 << 32) | (unsigned int)v9);
      v15 = *(unsigned int *)(v11 + 8);
      if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(v11 + 12) )
      {
        sub_C8D5F0(v11, v11 + 16, v15 + 1, 8);
        v15 = *(unsigned int *)(v11 + 8);
      }
      *(_QWORD *)(*(_QWORD *)v11 + 8 * v15) = v14;
      ++*(_DWORD *)(v11 + 8);
      v16 = v7[1];
      v17 = *(_QWORD *)(a1 + 16);
      v18 = *(unsigned int *)(v17 + 8);
      if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(v17 + 12) )
      {
        sub_C8D5F0(v17, v17 + 16, v18 + 1, 8);
        v18 = *(unsigned int *)(v17 + 8);
      }
      v7 += 2;
      *(_QWORD *)(*(_QWORD *)v17 + 8 * v18) = v16;
    }
    sub_A1FB70(**(_QWORD **)(a1 + 24), a3, *(_QWORD *)(a1 + 16), 0);
    *(_DWORD *)(*(_QWORD *)(a1 + 16) + 8LL) = 0;
    *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) = 0;
  }
}
