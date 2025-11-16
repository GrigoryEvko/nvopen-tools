// Function: sub_2B73030
// Address: 0x2b73030
//
unsigned __int64 __fastcall sub_2B73030(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v5; // rdx
  unsigned __int64 v6; // rcx
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rax
  unsigned __int8 *v9; // rdi
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r13
  __int64 v14; // rdx
  unsigned int v15; // esi
  __int64 v16; // rcx
  int v17; // r9d
  unsigned int i; // eax
  unsigned int v19; // eax
  __int64 v20; // rdi
  unsigned __int64 v21; // rax
  __int64 v22; // r9
  __int64 *v23; // r8
  unsigned int j; // eax
  __int64 *v25; // rbx
  __int64 v26; // r11
  int v27; // eax
  __int64 v28; // rax
  _QWORD *v29; // rax
  unsigned __int64 v30; // rsi
  unsigned __int64 v31; // rax
  __int64 *v33; // r14
  unsigned __int64 v34; // rax
  __int64 v35; // r13
  unsigned __int64 v36; // rdx
  unsigned __int64 v37; // rdx
  __int64 *v38; // r8
  __int64 *v39; // r14
  unsigned __int8 *v40; // r13
  int v41; // edx
  __int64 v42; // rax
  int v43; // eax
  __int64 v44; // [rsp+8h] [rbp-98h]
  __int64 v45; // [rsp+10h] [rbp-90h]
  __int64 *v46; // [rsp+18h] [rbp-88h]
  unsigned __int8 *v47; // [rsp+20h] [rbp-80h]
  unsigned __int8 *v48; // [rsp+28h] [rbp-78h]
  __int64 *v49; // [rsp+38h] [rbp-68h]
  unsigned __int8 *v50; // [rsp+38h] [rbp-68h]
  __int64 v51; // [rsp+40h] [rbp-60h] BYREF
  __int64 *v52; // [rsp+48h] [rbp-58h] BYREF
  unsigned __int64 v53; // [rsp+50h] [rbp-50h] BYREF
  unsigned __int8 *v54; // [rsp+58h] [rbp-48h]
  char v55; // [rsp+60h] [rbp-40h]

  v5 = *(_QWORD *)(a3 + 40);
  v51 = a2;
  v6 = HIDWORD(v5);
  v7 = 0x9DDFEA08EB382D69LL * (HIDWORD(v5) ^ (((8 * v5) & 0x7FFFFFFF8LL) + 12995744));
  v53 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL * ((v7 >> 47) ^ v6 ^ v7)) >> 47) ^ (0x9DDFEA08EB382D69LL * ((v7 >> 47) ^ v6 ^ v7)));
  v8 = sub_2B3B820((__int64 *)&v53, &v51);
  v9 = *(unsigned __int8 **)(a3 - 32);
  v51 = v8;
  v47 = sub_98ACB0(v9, qword_500FC48);
  sub_2B727F0((__int64)&v53, *a1, (unsigned __int64 *)&v51, v10, v11, v12);
  v13 = a1[1];
  v14 = v51;
  if ( v55 )
  {
    v15 = *(_DWORD *)(v13 + 24);
  }
  else
  {
    v15 = *(_DWORD *)(v13 + 24);
    v16 = *(_QWORD *)(v13 + 8);
    if ( v15 )
    {
      v17 = 1;
      for ( i = (v15 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((((0xBF58476D1CE4E5B9LL * v51) >> 31) ^ (0xBF58476D1CE4E5B9LL * v51)) << 32)
                  | ((unsigned int)v47 >> 4) ^ ((unsigned int)v47 >> 9))) >> 31)
               ^ (484763065 * (((unsigned int)v47 >> 4) ^ ((unsigned int)v47 >> 9)))); ; i = (v15 - 1) & v19 )
      {
        v44 = v16 + 80LL * i;
        if ( v51 == *(_QWORD *)v44 && v47 == *(unsigned __int8 **)(v44 + 8) )
          break;
        if ( *(_QWORD *)v44 == -1 && *(_QWORD *)(v44 + 8) == -4096 )
          goto LABEL_8;
        v19 = v17 + i;
        ++v17;
      }
      if ( v44 != 80LL * v15 + v16 )
      {
        v33 = *(__int64 **)(v44 + 16);
        v34 = *(unsigned int *)(v44 + 24);
        v49 = &v33[v34];
        if ( v33 != v49 )
        {
          do
          {
            v35 = *v33;
            v53 = sub_D35010(
                    *(_QWORD *)(*v33 + 8),
                    *(_QWORD *)(*v33 - 32),
                    *(_QWORD *)(a3 + 8),
                    *(_QWORD *)(a3 - 32),
                    a1[2],
                    a1[3],
                    1,
                    1);
            if ( BYTE4(v53) )
            {
              v36 = *(_QWORD *)(v35 - 32);
              v37 = 0x9DDFEA08EB382D69LL
                  * (((0x9DDFEA08EB382D69LL * (HIDWORD(v36) ^ (((8 * v36) & 0x7FFFFFFF8LL) + 12995744))) >> 47)
                   ^ (0x9DDFEA08EB382D69LL * (HIDWORD(v36) ^ (((8 * v36) & 0x7FFFFFFF8LL) + 12995744)))
                   ^ HIDWORD(v36));
              return 0x9DDFEA08EB382D69LL * (v37 ^ (v37 >> 47));
            }
            ++v33;
          }
          while ( v49 != v33 );
          v38 = *(__int64 **)(v44 + 16);
          v34 = *(unsigned int *)(v44 + 24);
          v46 = &v38[v34];
          if ( v38 != v46 )
          {
            v39 = *(__int64 **)(v44 + 16);
            do
            {
              v40 = *(unsigned __int8 **)(a3 - 32);
              v45 = *v39;
              v48 = *(unsigned __int8 **)(*v39 - 32);
              v50 = sub_98ACB0(v48, qword_500FC48);
              if ( v50 == sub_98ACB0(v40, qword_500FC48) && sub_2B608C0((__int64)v48, (char *)v40) )
                return sub_22AE640(*(_QWORD *)(v45 - 32));
              ++v39;
            }
            while ( v46 != v39 );
            v34 = *(unsigned int *)(v44 + 24);
          }
        }
        if ( v34 > 2 )
          return sub_22AE640(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v44 + 16) + 8 * v34 - 8) - 32LL));
        v13 = a1[1];
        v14 = v51;
        v15 = *(_DWORD *)(v13 + 24);
      }
    }
  }
LABEL_8:
  v53 = v14;
  v54 = v47;
  if ( !v15 )
  {
    v52 = 0;
    ++*(_QWORD *)v13;
    goto LABEL_39;
  }
  v21 = 0xBF58476D1CE4E5B9LL
      * (((((0xBF58476D1CE4E5B9LL * v14) >> 31) ^ (0xBF58476D1CE4E5B9LL * v14)) << 32)
       | ((unsigned int)v47 >> 4) ^ ((unsigned int)v47 >> 9));
  v22 = 1;
  v23 = 0;
  for ( j = (v15 - 1) & ((v21 >> 31) ^ v21); ; j = (v15 - 1) & v27 )
  {
    v20 = *(_QWORD *)(v13 + 8);
    v25 = (__int64 *)(v20 + 80LL * j);
    v26 = *v25;
    if ( *v25 == v14 && v47 == (unsigned __int8 *)v25[1] )
    {
      v28 = *((unsigned int *)v25 + 6);
      if ( v28 + 1 > (unsigned __int64)*((unsigned int *)v25 + 7) )
      {
        sub_C8D5F0((__int64)(v25 + 2), v25 + 4, v28 + 1, 8u, (__int64)v23, v22);
        v29 = (_QWORD *)(v25[2] + 8LL * *((unsigned int *)v25 + 6));
      }
      else
      {
        v29 = (_QWORD *)(v25[2] + 8 * v28);
      }
      goto LABEL_20;
    }
    if ( v26 == -1 )
      break;
    if ( v26 == -2 && v25[1] == -8192 && !v23 )
      v23 = (__int64 *)(v20 + 80LL * j);
LABEL_16:
    v27 = v22 + j;
    v22 = (unsigned int)(v22 + 1);
  }
  if ( v25[1] != -4096 )
    goto LABEL_16;
  v43 = *(_DWORD *)(v13 + 16);
  if ( !v23 )
    v23 = v25;
  v41 = v43 + 1;
  v52 = v23;
  ++*(_QWORD *)v13;
  if ( 4 * (v43 + 1) >= 3 * v15 )
  {
LABEL_39:
    v15 *= 2;
    goto LABEL_40;
  }
  if ( v15 - *(_DWORD *)(v13 + 20) - v41 > v15 >> 3 )
    goto LABEL_41;
LABEL_40:
  sub_2B54F00(v13, v15);
  sub_2B3F050(v13, (__int64 *)&v53, &v52);
  v41 = *(_DWORD *)(v13 + 16) + 1;
LABEL_41:
  v25 = v52;
  *(_DWORD *)(v13 + 16) = v41;
  if ( *v25 != -1 || v25[1] != -4096 )
    --*(_DWORD *)(v13 + 20);
  *v25 = v53;
  v42 = (__int64)v54;
  v25[3] = 0x600000000LL;
  v25[1] = v42;
  v29 = v25 + 4;
  v25[2] = (__int64)(v25 + 4);
LABEL_20:
  *v29 = a3;
  ++*((_DWORD *)v25 + 6);
  v30 = *(_QWORD *)(a3 - 32);
  v31 = 0x9DDFEA08EB382D69LL
      * (HIDWORD(v30)
       ^ (0x9DDFEA08EB382D69LL * (HIDWORD(v30) ^ (((8 * v30) & 0x7FFFFFFF8LL) + 12995744)))
       ^ ((0x9DDFEA08EB382D69LL * (HIDWORD(v30) ^ (((8 * v30) & 0x7FFFFFFF8LL) + 12995744))) >> 47));
  return 0x9DDFEA08EB382D69LL * ((v31 >> 47) ^ v31);
}
