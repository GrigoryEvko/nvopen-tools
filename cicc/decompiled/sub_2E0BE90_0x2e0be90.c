// Function: sub_2E0BE90
// Address: 0x2e0be90
//
__int64 __fastcall sub_2E0BE90(_DWORD *a1, __int64 a2)
{
  __int64 v2; // r15
  unsigned int **v4; // rbx
  unsigned int *v5; // r14
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 *v8; // rdx
  unsigned int *v9; // r12
  unsigned __int64 v10; // rcx
  unsigned int *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // rbx
  __int64 v18; // r15
  signed __int64 v19; // r14
  __int64 *v20; // rsi
  unsigned int *v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rsi
  _QWORD *v24; // r9
  unsigned int v25; // r10d
  __int64 v26; // rdi
  __int64 *v27; // rcx
  unsigned int **v28; // [rsp+8h] [rbp-58h]
  unsigned int *v29; // [rsp+10h] [rbp-50h]
  unsigned int **v30; // [rsp+18h] [rbp-48h]
  _DWORD *v31; // [rsp+20h] [rbp-40h]
  __int64 v32; // [rsp+28h] [rbp-38h]

  v2 = a2;
  a1[4] = 0;
  a1[14] = 0;
  v31 = a1 + 2;
  sub_3157150(a1 + 2, *(unsigned int *)(a2 + 72));
  v4 = *(unsigned int ***)(a2 + 64);
  v29 = 0;
  v30 = &v4[*(unsigned int *)(a2 + 72)];
  if ( v4 == v30 )
    goto LABEL_14;
  v5 = 0;
  do
  {
    while ( 1 )
    {
      v9 = *v4;
      v10 = *((_QWORD *)*v4 + 1) & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v10 )
        break;
      v6 = (*((__int64 *)*v4 + 1) >> 1) & 3;
      if ( (_DWORD)v6 )
      {
        v7 = v10 | (2LL * ((int)v6 - 1));
        v8 = (__int64 *)sub_2E09D00((__int64 *)v2, v7);
        if ( v8 != (__int64 *)(*(_QWORD *)v2 + 24LL * *(unsigned int *)(v2 + 8))
          && (*(_DWORD *)((*v8 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v8 >> 1) & 3) <= (*(_DWORD *)((v7 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                               | (unsigned int)(v7 >> 1)
                                                                                               & 3) )
        {
          v12 = (unsigned int *)v8[2];
          if ( v12 )
          {
            v5 = v9;
            sub_31571F0(v31, *v9, *v12);
            goto LABEL_7;
          }
        }
      }
      else
      {
        v13 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
        v14 = *(_QWORD *)(v10 + 16);
        if ( v14 )
        {
          v15 = *(_QWORD *)(v14 + 24);
        }
        else
        {
          v23 = *(unsigned int *)(v13 + 304);
          v24 = *(_QWORD **)(v13 + 296);
          if ( *(_DWORD *)(v13 + 304) )
          {
            v25 = *(_DWORD *)(v10 + 24);
            do
            {
              while ( 1 )
              {
                v26 = v23 >> 1;
                v27 = &v24[2 * (v23 >> 1)];
                if ( v25 < (*(_DWORD *)((*v27 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v27 >> 1) & 3) )
                  break;
                v24 = v27 + 2;
                v23 = v23 - v26 - 1;
                if ( v23 <= 0 )
                  goto LABEL_37;
              }
              v23 >>= 1;
            }
            while ( v26 > 0 );
          }
LABEL_37:
          v15 = *(v24 - 1);
        }
        v16 = *(_QWORD *)(v15 + 64);
        v32 = v16 + 8LL * *(unsigned int *)(v15 + 72);
        if ( v16 != v32 )
        {
          v28 = v4;
          v17 = v2;
          v18 = *(_QWORD *)(v15 + 64);
          while ( 1 )
          {
            v22 = *(_QWORD *)(*(_QWORD *)(v13 + 152) + 16LL * *(unsigned int *)(*(_QWORD *)v18 + 24LL) + 8);
            if ( ((v22 >> 1) & 3) != 0 )
              v19 = v22 & 0xFFFFFFFFFFFFFFF8LL | (2LL * (int)(((v22 >> 1) & 3) - 1));
            else
              v19 = *(_QWORD *)(v22 & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL | 6;
            v20 = (__int64 *)sub_2E09D00((__int64 *)v17, v19);
            if ( v20 != (__int64 *)(*(_QWORD *)v17 + 24LL * *(unsigned int *)(v17 + 8))
              && (*(_DWORD *)((*v20 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v20 >> 1) & 3) <= (*(_DWORD *)((v19 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v19 >> 1) & 3) )
            {
              v21 = (unsigned int *)v20[2];
              if ( v21 )
                sub_31571F0(v31, *v9, *v21);
            }
            v18 += 8;
            if ( v32 == v18 )
              break;
            v13 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
          }
          v2 = v17;
          v4 = v28;
        }
      }
      v5 = v9;
LABEL_7:
      if ( v30 == ++v4 )
        goto LABEL_11;
    }
    if ( !v29 )
    {
      v29 = *v4;
      goto LABEL_7;
    }
    ++v4;
    sub_31571F0(v31, *v29, *v9);
    v29 = v9;
  }
  while ( v30 != v4 );
LABEL_11:
  if ( v5 && v29 )
    sub_31571F0(v31, *v5, *v29);
LABEL_14:
  sub_3157250(v31);
  return (unsigned int)a1[14];
}
