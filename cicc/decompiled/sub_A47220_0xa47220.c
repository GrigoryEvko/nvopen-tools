// Function: sub_A47220
// Address: 0xa47220
//
void __fastcall sub_A47220(__int64 a1, __int64 a2)
{
  unsigned int v2; // r8d
  __int64 v4; // r9
  int v6; // r12d
  __int64 v7; // rdi
  __int64 *v8; // rcx
  unsigned int v9; // esi
  __int64 *v10; // rdx
  __int64 v11; // r11
  _DWORD *v12; // r12
  int v13; // eax
  unsigned __int32 v14; // r12d
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned int v17; // esi
  __int64 v18; // rdi
  int v19; // r13d
  __int64 v20; // r8
  unsigned __int64 v21; // rax
  unsigned int *v22; // r9
  unsigned int i; // eax
  unsigned int *v24; // r10
  __int64 v25; // rcx
  _DWORD *v26; // r13
  int v27; // r8d
  int v28; // r8d
  unsigned int *v29; // r11
  __int64 v30; // rdi
  int v31; // r13d
  unsigned int k; // eax
  __int64 v33; // rax
  __m128i *v34; // rsi
  __m128i *v35; // rsi
  __int64 v36; // rsi
  __int64 v37; // r13
  __int64 v38; // rax
  __int64 *v39; // r12
  __int64 *v40; // r15
  char *v41; // rax
  int v42; // eax
  int v43; // esi
  __int64 v44; // r9
  unsigned int v45; // eax
  int v46; // edx
  __int64 v47; // r8
  __int64 v48; // rax
  char *v49; // rsi
  char *v50; // rsi
  unsigned int v51; // eax
  unsigned int v52; // eax
  int v53; // eax
  int v54; // eax
  int v55; // esi
  int v56; // r11d
  __int64 *v57; // r10
  __int64 v58; // r9
  unsigned int v59; // eax
  __int64 v60; // r8
  int v61; // eax
  int v62; // r8d
  int v63; // r8d
  __int64 v64; // rdi
  int v65; // r13d
  unsigned int j; // eax
  unsigned int v67; // eax
  int v68; // r11d
  __int64 v69; // [rsp+0h] [rbp-60h]
  __int64 v70; // [rsp+8h] [rbp-58h] BYREF
  __int64 v71; // [rsp+10h] [rbp-50h] BYREF
  __int64 v72; // [rsp+18h] [rbp-48h] BYREF
  __m128i v73; // [rsp+20h] [rbp-40h] BYREF

  v70 = a2;
  if ( !a2 )
    return;
  v2 = *(_DWORD *)(a1 + 408);
  v4 = a1 + 384;
  if ( v2 )
  {
    v6 = 1;
    v7 = *(_QWORD *)(a1 + 392);
    v8 = 0;
    v9 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v10 = (__int64 *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
    {
LABEL_4:
      v12 = v10 + 1;
      if ( *((_DWORD *)v10 + 2) )
        goto LABEL_5;
      goto LABEL_50;
    }
    while ( v11 != -4 )
    {
      if ( v11 == -8 && !v8 )
        v8 = v10;
      v9 = (v2 - 1) & (v6 + v9);
      v10 = (__int64 *)(v7 + 16LL * v9);
      v11 = *v10;
      if ( a2 == *v10 )
        goto LABEL_4;
      ++v6;
    }
    v53 = *(_DWORD *)(a1 + 400);
    if ( !v8 )
      v8 = v10;
    ++*(_QWORD *)(a1 + 384);
    v46 = v53 + 1;
    if ( 4 * (v53 + 1) < 3 * v2 )
    {
      if ( v2 - *(_DWORD *)(a1 + 404) - v46 > v2 >> 3 )
        goto LABEL_47;
      sub_A46D90(v4, v2);
      v54 = *(_DWORD *)(a1 + 408);
      if ( v54 )
      {
        v55 = v54 - 1;
        v56 = 1;
        v57 = 0;
        v58 = *(_QWORD *)(a1 + 392);
        v59 = (v54 - 1) & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
        v46 = *(_DWORD *)(a1 + 400) + 1;
        v8 = (__int64 *)(v58 + 16LL * v59);
        v60 = *v8;
        if ( *v8 != v70 )
        {
          while ( v60 != -4 )
          {
            if ( v60 == -8 && !v57 )
              v57 = v8;
            v59 = v55 & (v56 + v59);
            v8 = (__int64 *)(v58 + 16LL * v59);
            v60 = *v8;
            if ( v70 == *v8 )
              goto LABEL_47;
            ++v56;
          }
LABEL_77:
          if ( v57 )
            v8 = v57;
          goto LABEL_47;
        }
        goto LABEL_47;
      }
LABEL_113:
      ++*(_DWORD *)(a1 + 400);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 384);
  }
  sub_A46D90(v4, 2 * v2);
  v42 = *(_DWORD *)(a1 + 408);
  if ( !v42 )
    goto LABEL_113;
  v43 = v42 - 1;
  v44 = *(_QWORD *)(a1 + 392);
  v45 = (v42 - 1) & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
  v46 = *(_DWORD *)(a1 + 400) + 1;
  v8 = (__int64 *)(v44 + 16LL * v45);
  v47 = *v8;
  if ( *v8 != v70 )
  {
    v68 = 1;
    v57 = 0;
    while ( v47 != -4 )
    {
      if ( !v57 && v47 == -8 )
        v57 = v8;
      v45 = v43 & (v68 + v45);
      v8 = (__int64 *)(v44 + 16LL * v45);
      v47 = *v8;
      if ( v70 == *v8 )
        goto LABEL_47;
      ++v68;
    }
    goto LABEL_77;
  }
LABEL_47:
  *(_DWORD *)(a1 + 400) = v46;
  if ( *v8 != -4 )
    --*(_DWORD *)(a1 + 404);
  v48 = v70;
  *((_DWORD *)v8 + 2) = 0;
  v12 = v8 + 1;
  *v8 = v48;
LABEL_50:
  v49 = *(char **)(a1 + 424);
  if ( v49 == *(char **)(a1 + 432) )
  {
    sub_A40410((char **)(a1 + 416), v49, &v70);
    v50 = *(char **)(a1 + 424);
  }
  else
  {
    if ( v49 )
    {
      *(_QWORD *)v49 = v70;
      v49 = *(char **)(a1 + 424);
    }
    v50 = v49 + 8;
    *(_QWORD *)(a1 + 424) = v50;
  }
  *v12 = (__int64)&v50[-*(_QWORD *)(a1 + 416)] >> 3;
LABEL_5:
  v13 = sub_A74480(&v70);
  HIDWORD(v69) = v13 - 1;
  if ( v13 )
  {
    v14 = -1;
    do
    {
      while ( 1 )
      {
        v15 = sub_A74490(&v70, v14);
        v71 = v15;
        v16 = v15;
        if ( v15 )
        {
          v17 = *(_DWORD *)(a1 + 352);
          v73.m128i_i32[0] = v14;
          v18 = a1 + 328;
          v73.m128i_i64[1] = v15;
          if ( v17 )
          {
            v19 = 1;
            v20 = *(_QWORD *)(a1 + 336);
            v21 = 0xBF58476D1CE4E5B9LL
                * (((unsigned __int64)(37 * v14) << 32) | ((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
            v22 = 0;
            for ( i = (v17 - 1) & ((v21 >> 31) ^ v21); ; i = (v17 - 1) & v51 )
            {
              v24 = (unsigned int *)(v20 + 24LL * i);
              v25 = *v24;
              if ( (_DWORD)v25 == v14 && v16 == *((_QWORD *)v24 + 1) )
              {
                v26 = v24 + 4;
                if ( !v24[4] )
                  goto LABEL_33;
                goto LABEL_19;
              }
              if ( (_DWORD)v25 == -1 )
              {
                if ( *((_QWORD *)v24 + 1) == -4 )
                {
                  v61 = *(_DWORD *)(a1 + 344);
                  if ( !v22 )
                    v22 = v24;
                  ++*(_QWORD *)(a1 + 328);
                  v20 = (unsigned int)(v61 + 1);
                  if ( 4 * (int)v20 >= 3 * v17 )
                    goto LABEL_22;
                  v16 = v14;
                  if ( v17 - *(_DWORD *)(a1 + 348) - (unsigned int)v20 <= v17 >> 3 )
                  {
                    sub_A46F70(v18, v17);
                    v62 = *(_DWORD *)(a1 + 352);
                    if ( v62 )
                    {
                      v16 = v73.m128i_u32[0];
                      v63 = v62 - 1;
                      v29 = 0;
                      v65 = 1;
                      for ( j = v63
                              & (((0xBF58476D1CE4E5B9LL
                                 * (((unsigned __int64)(unsigned int)(37 * v73.m128i_i32[0]) << 32)
                                  | ((unsigned __int32)v73.m128i_i32[2] >> 9)
                                  ^ ((unsigned __int32)v73.m128i_i32[2] >> 4))) >> 31)
                               ^ (484763065
                                * (((unsigned __int32)v73.m128i_i32[2] >> 9) ^ ((unsigned __int32)v73.m128i_i32[2] >> 4))));
                            ;
                            j = v63 & v67 )
                      {
                        v64 = *(_QWORD *)(a1 + 336);
                        v22 = (unsigned int *)(v64 + 24LL * j);
                        v25 = *v22;
                        if ( (_DWORD)v25 == v73.m128i_i32[0] && *((_QWORD *)v22 + 1) == v73.m128i_i64[1] )
                          break;
                        if ( (_DWORD)v25 == -1 )
                        {
                          if ( *((_QWORD *)v22 + 1) == -4 )
                            goto LABEL_27;
                        }
                        else if ( (_DWORD)v25 == -2 && *((_QWORD *)v22 + 1) == -8 && !v29 )
                        {
                          v29 = (unsigned int *)(v64 + 24LL * j);
                        }
                        v67 = v65 + j;
                        ++v65;
                      }
                      goto LABEL_83;
                    }
LABEL_114:
                    ++*(_DWORD *)(a1 + 344);
                    BUG();
                  }
                  goto LABEL_30;
                }
              }
              else if ( (_DWORD)v25 == -2 && *((_QWORD *)v24 + 1) == -8 && !v22 )
              {
                v22 = (unsigned int *)(v20 + 24LL * i);
              }
              v51 = v19 + i;
              ++v19;
            }
          }
          ++*(_QWORD *)(a1 + 328);
LABEL_22:
          sub_A46F70(v18, 2 * v17);
          v27 = *(_DWORD *)(a1 + 352);
          if ( !v27 )
            goto LABEL_114;
          v16 = v73.m128i_u32[0];
          v28 = v27 - 1;
          v29 = 0;
          v31 = 1;
          for ( k = v28
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned __int64)(unsigned int)(37 * v73.m128i_i32[0]) << 32)
                      | ((unsigned __int32)v73.m128i_i32[2] >> 9) ^ ((unsigned __int32)v73.m128i_i32[2] >> 4))) >> 31)
                   ^ (484763065 * (((unsigned __int32)v73.m128i_i32[2] >> 9) ^ ((unsigned __int32)v73.m128i_i32[2] >> 4))));
                ;
                k = v28 & v52 )
          {
            v30 = *(_QWORD *)(a1 + 336);
            v22 = (unsigned int *)(v30 + 24LL * k);
            v25 = *v22;
            if ( (_DWORD)v25 == v73.m128i_i32[0] && *((_QWORD *)v22 + 1) == v73.m128i_i64[1] )
              break;
            if ( (_DWORD)v25 == -1 )
            {
              if ( *((_QWORD *)v22 + 1) == -4 )
              {
LABEL_27:
                if ( v29 )
                  v22 = v29;
                v20 = (unsigned int)(*(_DWORD *)(a1 + 344) + 1);
                goto LABEL_30;
              }
            }
            else if ( (_DWORD)v25 == -2 && *((_QWORD *)v22 + 1) == -8 && !v29 )
            {
              v29 = (unsigned int *)(v30 + 24LL * k);
            }
            v52 = v31 + k;
            ++v31;
          }
LABEL_83:
          v20 = (unsigned int)(*(_DWORD *)(a1 + 344) + 1);
LABEL_30:
          *(_DWORD *)(a1 + 344) = v20;
          if ( *v22 != -1 || *((_QWORD *)v22 + 1) != -4 )
            --*(_DWORD *)(a1 + 348);
          *v22 = v16;
          v33 = v73.m128i_i64[1];
          v26 = v22 + 4;
          v22[4] = 0;
          *((_QWORD *)v22 + 1) = v33;
LABEL_33:
          v34 = *(__m128i **)(a1 + 368);
          if ( v34 == *(__m128i **)(a1 + 376) )
          {
            sub_A40590((const __m128i **)(a1 + 360), v34, &v73);
            v35 = *(__m128i **)(a1 + 368);
          }
          else
          {
            if ( v34 )
            {
              *v34 = _mm_loadu_si128(&v73);
              v34 = *(__m128i **)(a1 + 368);
            }
            v35 = v34 + 1;
            *(_QWORD *)(a1 + 368) = v35;
          }
          v36 = ((__int64)v35->m128i_i64 - *(_QWORD *)(a1 + 360)) >> 4;
          *v26 = v36;
          v37 = ((__int64 (__fastcall *)(__int64 *, __int64, __int64, __int64, __int64, unsigned int *, __int64, __int64))sub_A73280)(
                  &v71,
                  v36,
                  v16,
                  v25,
                  v20,
                  v22,
                  v69,
                  v70);
          v38 = sub_A73290(&v71);
          if ( v37 != v38 )
            break;
        }
LABEL_19:
        if ( HIDWORD(v69) == ++v14 )
          return;
      }
      LODWORD(v69) = v14;
      v39 = (__int64 *)v37;
      v40 = (__int64 *)v38;
      do
      {
        while ( 1 )
        {
          v72 = *v39;
          if ( (unsigned __int8)sub_A71860(&v72) )
            break;
          if ( v40 == ++v39 )
            goto LABEL_42;
        }
        ++v39;
        v41 = (char *)sub_A72A60(&v72);
        sub_A44BF0(a1, v41);
      }
      while ( v40 != v39 );
LABEL_42:
      v14 = v69 + 1;
    }
    while ( HIDWORD(v69) != (_DWORD)v69 + 1 );
  }
}
