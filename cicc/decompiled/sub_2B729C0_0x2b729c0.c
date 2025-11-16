// Function: sub_2B729C0
// Address: 0x2b729c0
//
unsigned __int64 __fastcall sub_2B729C0(__int64 *a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  unsigned __int64 v5; // rdx
  unsigned __int64 v6; // rcx
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rax
  unsigned __int8 *v9; // rdi
  unsigned __int8 *v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r8
  _QWORD *v13; // r9
  __int64 v14; // rsi
  _QWORD *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdi
  int v19; // r11d
  unsigned int i; // eax
  unsigned int v21; // eax
  __int64 v22; // r12
  unsigned int v23; // esi
  __int64 v24; // rdi
  unsigned __int64 v25; // rax
  __int64 v26; // r9
  __int64 *v27; // r8
  unsigned int j; // eax
  __int64 *v29; // rbx
  __int64 v30; // r10
  int v31; // eax
  _QWORD *v32; // rax
  __int64 v33; // rax
  _QWORD *v34; // rax
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // rax
  __int64 *v38; // r12
  unsigned __int64 v39; // rax
  __int64 *v40; // r14
  __int64 v41; // r13
  unsigned __int64 v42; // rdi
  unsigned __int64 v43; // rdx
  __int64 *v44; // r12
  __int64 v45; // r14
  unsigned __int8 *v46; // r13
  unsigned __int8 *v47; // r15
  int v48; // edx
  __int64 v49; // rax
  int v50; // eax
  __int64 v51; // [rsp-8h] [rbp-A8h]
  __int64 v52; // [rsp+0h] [rbp-A0h]
  __int64 *v53; // [rsp+8h] [rbp-98h]
  __int64 v54; // [rsp+20h] [rbp-80h]
  unsigned __int8 *v55; // [rsp+30h] [rbp-70h]
  unsigned __int8 *v56; // [rsp+38h] [rbp-68h]
  unsigned __int64 v57; // [rsp+40h] [rbp-60h] BYREF
  __int64 *v58; // [rsp+48h] [rbp-58h] BYREF
  __int64 v59; // [rsp+50h] [rbp-50h] BYREF
  unsigned __int8 *v60; // [rsp+58h] [rbp-48h]

  v3 = a3;
  v5 = *(_QWORD *)(a3 + 40);
  v57 = a2;
  v6 = HIDWORD(v5);
  v7 = 0x9DDFEA08EB382D69LL * (HIDWORD(v5) ^ (((8 * v5) & 0x7FFFFFFF8LL) + 12995744));
  v59 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL * ((v7 >> 47) ^ v6 ^ v7)) >> 47) ^ (0x9DDFEA08EB382D69LL * ((v7 >> 47) ^ v6 ^ v7)));
  v8 = sub_2B3B820(&v59, (__int64 *)&v57);
  v9 = *(unsigned __int8 **)(v3 - 32);
  v57 = v8;
  v10 = sub_98ACB0(v9, qword_500FC48);
  v14 = *a1;
  v55 = v10;
  if ( *(_QWORD *)(*a1 + 72) )
  {
    v32 = *(_QWORD **)(v14 + 48);
    v13 = (_QWORD *)(v14 + 40);
    if ( !v32 )
      goto LABEL_14;
    v12 = v14 + 40;
    do
    {
      v11 = v32[2];
      if ( v32[4] < v57 )
      {
        v32 = (_QWORD *)v32[3];
      }
      else
      {
        v12 = (__int64)v32;
        v32 = (_QWORD *)v32[2];
      }
    }
    while ( v32 );
    if ( (_QWORD *)v12 == v13 || v57 < *(_QWORD *)(v12 + 32) )
      goto LABEL_14;
    goto LABEL_7;
  }
  v15 = *(_QWORD **)v14;
  v16 = *(_QWORD *)v14 + 8LL * *(unsigned int *)(v14 + 8);
  if ( *(_QWORD *)v14 != v16 )
  {
    v11 = v57;
    while ( *v15 != v57 )
    {
      if ( (_QWORD *)v16 == ++v15 )
        goto LABEL_14;
    }
    if ( (_QWORD *)v16 != v15 )
    {
LABEL_7:
      v17 = a1[1];
      v12 = v57;
      v11 = *(unsigned int *)(v17 + 24);
      v18 = *(_QWORD *)(v17 + 8);
      if ( (_DWORD)v11 )
      {
        v19 = 1;
        for ( i = (v11 - 1)
                & (((0xBF58476D1CE4E5B9LL
                   * (((((0xBF58476D1CE4E5B9LL * v57) >> 31) ^ (0xBF58476D1CE4E5B9LL * v57)) << 32)
                    | ((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4))) >> 31)
                 ^ (484763065 * (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4)))); ; i = (v11 - 1) & v21 )
        {
          v13 = (_QWORD *)(v18 + 80LL * i);
          if ( v57 == *v13 && v55 == (unsigned __int8 *)v13[1] )
            break;
          if ( *v13 == -1 && v13[1] == -4096 )
            goto LABEL_14;
          v21 = v19 + i;
          ++v19;
        }
        v52 = v18 + 80LL * i;
        if ( v13 != (_QWORD *)(80 * v11 + v18) )
        {
          v38 = *(__int64 **)(v52 + 16);
          v39 = *(unsigned int *)(v52 + 24);
          v40 = &v38[v39];
          if ( v38 != v40 )
          {
            do
            {
              v41 = *v38;
              v59 = sub_D35010(
                      *(_QWORD *)(*v38 + 8),
                      *(_QWORD *)(*v38 - 32),
                      *(_QWORD *)(v3 + 8),
                      *(_QWORD *)(v3 - 32),
                      *(_QWORD *)(a1[2] + 3344),
                      *(_QWORD *)(a1[2] + 3288),
                      1,
                      1);
              v11 = v51;
              if ( BYTE4(v59) )
              {
                v42 = *(_QWORD *)(v41 - 32);
                v43 = 0x9DDFEA08EB382D69LL
                    * (HIDWORD(v42)
                     ^ (0x9DDFEA08EB382D69LL * (HIDWORD(v42) ^ (((8 * v42) & 0x7FFFFFFF8LL) + 12995744)))
                     ^ ((0x9DDFEA08EB382D69LL * (HIDWORD(v42) ^ (((8 * v42) & 0x7FFFFFFF8LL) + 12995744))) >> 47));
                return 0x9DDFEA08EB382D69LL * (v43 ^ (v43 >> 47));
              }
              ++v38;
            }
            while ( v40 != v38 );
            v12 = *(_QWORD *)(v52 + 16);
            v39 = *(unsigned int *)(v52 + 24);
            v53 = (__int64 *)(v12 + 8 * v39);
            if ( (__int64 *)v12 != v53 )
            {
              v54 = v3;
              v44 = *(__int64 **)(v52 + 16);
              do
              {
                v45 = *v44;
                v46 = *(unsigned __int8 **)(*v44 - 32);
                v47 = *(unsigned __int8 **)(v54 - 32);
                v56 = sub_98ACB0(v46, qword_500FC48);
                if ( v56 == sub_98ACB0(v47, qword_500FC48) && sub_2B608C0((__int64)v46, (char *)v47) )
                  return sub_22AE640(*(_QWORD *)(v45 - 32));
                ++v44;
              }
              while ( v53 != v44 );
              v3 = v54;
              v39 = *(unsigned int *)(v52 + 24);
            }
          }
          if ( v39 > 2 )
            return sub_22AE640(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v52 + 16) + 8 * v39 - 8) - 32LL));
          v14 = *a1;
        }
      }
    }
  }
LABEL_14:
  sub_2B727F0((__int64)&v59, v14, &v57, v11, v12, (__int64)v13);
  v22 = a1[1];
  v59 = v57;
  v60 = v55;
  v23 = *(_DWORD *)(v22 + 24);
  if ( !v23 )
  {
    v58 = 0;
    ++*(_QWORD *)v22;
    goto LABEL_51;
  }
  v25 = 0xBF58476D1CE4E5B9LL
      * (((((0xBF58476D1CE4E5B9LL * v57) >> 31) ^ (0xBF58476D1CE4E5B9LL * v57)) << 32)
       | ((unsigned int)v55 >> 4) ^ ((unsigned int)v55 >> 9));
  v26 = 1;
  v27 = 0;
  for ( j = (v23 - 1) & ((v25 >> 31) ^ v25); ; j = (v23 - 1) & v31 )
  {
    v24 = *(_QWORD *)(v22 + 8);
    v29 = (__int64 *)(v24 + 80LL * j);
    v30 = *v29;
    if ( v57 == *v29 && v55 == (unsigned __int8 *)v29[1] )
    {
      v33 = *((unsigned int *)v29 + 6);
      if ( v33 + 1 > (unsigned __int64)*((unsigned int *)v29 + 7) )
      {
        sub_C8D5F0((__int64)(v29 + 2), v29 + 4, v33 + 1, 8u, (__int64)v27, v26);
        v34 = (_QWORD *)(v29[2] + 8LL * *((unsigned int *)v29 + 6));
      }
      else
      {
        v34 = (_QWORD *)(v29[2] + 8 * v33);
      }
      goto LABEL_32;
    }
    if ( v30 == -1 )
      break;
    if ( v30 == -2 && v29[1] == -8192 && !v27 )
      v27 = (__int64 *)(v24 + 80LL * j);
LABEL_22:
    v31 = v26 + j;
    v26 = (unsigned int)(v26 + 1);
  }
  if ( v29[1] != -4096 )
    goto LABEL_22;
  v50 = *(_DWORD *)(v22 + 16);
  if ( !v27 )
    v27 = v29;
  v48 = v50 + 1;
  v58 = v27;
  ++*(_QWORD *)v22;
  if ( 4 * (v50 + 1) >= 3 * v23 )
  {
LABEL_51:
    v23 *= 2;
    goto LABEL_52;
  }
  if ( v23 - *(_DWORD *)(v22 + 20) - v48 > v23 >> 3 )
    goto LABEL_53;
LABEL_52:
  sub_2B54F00(v22, v23);
  sub_2B3F050(v22, &v59, &v58);
  v48 = *(_DWORD *)(v22 + 16) + 1;
LABEL_53:
  v29 = v58;
  *(_DWORD *)(v22 + 16) = v48;
  if ( *v29 != -1 || v29[1] != -4096 )
    --*(_DWORD *)(v22 + 20);
  *v29 = v59;
  v49 = (__int64)v60;
  v29[3] = 0x600000000LL;
  v29[1] = v49;
  v34 = v29 + 4;
  v29[2] = (__int64)(v29 + 4);
LABEL_32:
  *v34 = v3;
  ++*((_DWORD *)v29 + 6);
  v35 = *(_QWORD *)(v3 - 32);
  v36 = 0x9DDFEA08EB382D69LL
      * ((0x9DDFEA08EB382D69LL * (HIDWORD(v35) ^ (((8 * v35) & 0x7FFFFFFF8LL) + 12995744)))
       ^ HIDWORD(v35)
       ^ ((0x9DDFEA08EB382D69LL * (HIDWORD(v35) ^ (((8 * v35) & 0x7FFFFFFF8LL) + 12995744))) >> 47));
  return 0x9DDFEA08EB382D69LL * ((v36 >> 47) ^ v36);
}
