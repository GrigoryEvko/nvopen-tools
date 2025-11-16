// Function: sub_3992C50
// Address: 0x3992c50
//
void __fastcall sub_3992C50(const __m128i *a1, unsigned int a2)
{
  char v3; // dl
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rax
  int v6; // r14d
  unsigned __int64 v7; // r15
  const __m128i *v8; // rax
  __m128i *v9; // r12
  unsigned __int32 v10; // r13d
  bool v11; // zf
  __int64 *v12; // r9
  __m128i *v13; // rax
  __int64 v14; // rdx
  __m128i *j; // rdx
  __int64 *k; // rdx
  __int64 v17; // rcx
  const __m128i *v18; // r8
  int v19; // esi
  __int64 v20; // rdi
  __int64 *v21; // r14
  __int64 v22; // r11
  int v23; // r13d
  unsigned __int64 v24; // r11
  unsigned __int64 v25; // r11
  unsigned int m; // eax
  __int64 *v27; // r11
  __int64 v28; // r15
  __int32 v29; // esi
  __int64 v30; // rax
  unsigned int v31; // eax
  __m128i *v32; // rax
  __m128i *v33; // rdx
  const __m128i *v34; // rcx
  __m128i *v35; // rdi
  const __m128i *v36; // r8
  int v37; // esi
  __int64 v38; // rdi
  int v39; // r14d
  __int64 *v40; // r11
  __int64 v41; // r9
  unsigned __int64 v42; // r9
  unsigned __int64 v43; // r9
  unsigned int i; // eax
  __int64 *v45; // r9
  __int64 v46; // r10
  unsigned int v47; // eax
  __int32 v48; // esi
  __int64 v49; // rax
  _BYTE v50[144]; // [rsp+10h] [rbp-90h] BYREF

  v3 = a1->m128i_i8[8] & 1;
  if ( a2 <= 3 )
  {
    if ( v3 )
      return;
    v4 = a1[1].m128i_u64[0];
    v10 = a1[1].m128i_u32[2];
    a1->m128i_i8[8] |= 1u;
  }
  else
  {
    v4 = a1[1].m128i_u64[0];
    v5 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
            | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
            | (a2 - 1)
            | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
          | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 16)
        | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
        | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1))
       + 1;
    v6 = v5;
    if ( (unsigned int)v5 > 0x40 )
    {
      v7 = 24LL * (unsigned int)v5;
      if ( v3 )
        goto LABEL_5;
      v10 = a1[1].m128i_u32[2];
    }
    else
    {
      if ( v3 )
      {
        v7 = 1536;
        v6 = 64;
LABEL_5:
        v8 = a1 + 1;
        v9 = (__m128i *)v50;
        while ( 1 )
        {
          if ( v8->m128i_i64[0] == -8 )
          {
            if ( v8->m128i_i64[1] != -8 )
              goto LABEL_7;
          }
          else if ( v8->m128i_i64[0] != -16 || v8->m128i_i64[1] != -16 )
          {
LABEL_7:
            if ( v9 )
              *v9 = _mm_loadu_si128(v8);
            v9 = (__m128i *)((char *)v9 + 24);
            v9[-1].m128i_i64[1] = v8[1].m128i_i64[0];
          }
          v8 = (const __m128i *)((char *)v8 + 24);
          if ( v8 == &a1[7] )
          {
            a1->m128i_i8[8] &= ~1u;
            v32 = (__m128i *)sub_22077B0(v7);
            a1[1].m128i_i32[2] = v6;
            v33 = (__m128i *)v50;
            v11 = (a1->m128i_i64[1] & 1) == 0;
            a1->m128i_i64[1] &= 1uLL;
            v34 = v32;
            a1[1].m128i_i64[0] = (__int64)v32;
            if ( !v11 )
            {
              v34 = a1 + 1;
              v32 = (__m128i *)&a1[1];
              v7 = 96;
            }
            v35 = (__m128i *)((char *)v32 + v7);
            while ( 1 )
            {
              if ( v34 )
              {
                v32->m128i_i64[0] = -8;
                v32->m128i_i64[1] = -8;
              }
              v32 = (__m128i *)((char *)v32 + 24);
              if ( v35 == v32 )
                break;
              v34 = v32;
            }
            while ( 2 )
            {
              if ( v9 == v33 )
                return;
              v17 = v33->m128i_i64[0];
              if ( v33->m128i_i64[0] == -8 )
              {
                if ( v33->m128i_i64[1] == -8 )
                {
LABEL_77:
                  v33 = (__m128i *)((char *)v33 + 24);
                  continue;
                }
              }
              else if ( v17 == -16 && v33->m128i_i64[1] == -16 )
              {
                goto LABEL_77;
              }
              break;
            }
            if ( (a1->m128i_i8[8] & 1) != 0 )
            {
              v36 = a1 + 1;
              v37 = 3;
            }
            else
            {
              v48 = a1[1].m128i_i32[2];
              v36 = (const __m128i *)a1[1].m128i_i64[0];
              if ( !v48 )
                goto LABEL_89;
              v37 = v48 - 1;
            }
            v38 = v33->m128i_i64[1];
            v39 = 1;
            v40 = 0;
            v41 = ((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4);
            v42 = (((v41 | ((unsigned __int64)(((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4)) << 32))
                  - 1
                  - (v41 << 32)) >> 22)
                ^ ((v41 | ((unsigned __int64)(((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4)) << 32))
                 - 1
                 - (v41 << 32));
            v43 = ((9 * (((v42 - 1 - (v42 << 13)) >> 8) ^ (v42 - 1 - (v42 << 13)))) >> 15)
                ^ (9 * (((v42 - 1 - (v42 << 13)) >> 8) ^ (v42 - 1 - (v42 << 13))));
            for ( i = v37 & (((v43 - 1 - (v43 << 27)) >> 31) ^ (v43 - 1 - ((_DWORD)v43 << 27))); ; i = v37 & v47 )
            {
              v45 = &v36->m128i_i64[3 * i];
              v46 = *v45;
              if ( v17 == *v45 && v45[1] == v38 )
                break;
              if ( v46 == -8 )
              {
                if ( v45[1] == -8 )
                {
                  if ( v40 )
                    v45 = v40;
                  break;
                }
              }
              else if ( v46 == -16 && v45[1] == -16 && !v40 )
              {
                v40 = &v36->m128i_i64[3 * i];
              }
              v47 = v39 + i;
              ++v39;
            }
            v49 = v33[1].m128i_i64[0];
            *v45 = v17;
            v45[1] = v38;
            v45[2] = v49;
            a1->m128i_i32[2] = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
            goto LABEL_77;
          }
        }
      }
      v10 = a1[1].m128i_u32[2];
      v7 = 1536;
      v6 = 64;
    }
    v30 = sub_22077B0(v7);
    a1[1].m128i_i32[2] = v6;
    a1[1].m128i_i64[0] = v30;
  }
  v11 = (a1->m128i_i64[1] & 1) == 0;
  a1->m128i_i64[1] &= 1uLL;
  v12 = (__int64 *)(v4 + 24LL * v10);
  if ( v11 )
  {
    v13 = (__m128i *)a1[1].m128i_i64[0];
    v14 = 24LL * a1[1].m128i_u32[2];
  }
  else
  {
    v13 = (__m128i *)&a1[1];
    v14 = 96;
  }
  for ( j = (__m128i *)((char *)v13 + v14); j != v13; v13 = (__m128i *)((char *)v13 + 24) )
  {
    if ( v13 )
    {
      v13->m128i_i64[0] = -8;
      v13->m128i_i64[1] = -8;
    }
  }
  for ( k = (__int64 *)v4; v12 != k; k += 3 )
  {
    v17 = *k;
    if ( *k == -8 )
    {
      if ( k[1] != -8 )
        goto LABEL_26;
    }
    else if ( v17 != -16 || k[1] != -16 )
    {
LABEL_26:
      if ( (a1->m128i_i8[8] & 1) != 0 )
      {
        v18 = a1 + 1;
        v19 = 3;
      }
      else
      {
        v29 = a1[1].m128i_i32[2];
        v18 = (const __m128i *)a1[1].m128i_i64[0];
        if ( !v29 )
        {
LABEL_89:
          MEMORY[0] = v17;
          BUG();
        }
        v19 = v29 - 1;
      }
      v20 = k[1];
      v21 = 0;
      v22 = ((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4);
      v23 = 1;
      v24 = (((v22 | ((unsigned __int64)(((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4)) << 32)) - 1 - (v22 << 32)) >> 22)
          ^ ((v22 | ((unsigned __int64)(((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4)) << 32)) - 1 - (v22 << 32));
      v25 = ((9 * (((v24 - 1 - (v24 << 13)) >> 8) ^ (v24 - 1 - (v24 << 13)))) >> 15)
          ^ (9 * (((v24 - 1 - (v24 << 13)) >> 8) ^ (v24 - 1 - (v24 << 13))));
      for ( m = v19 & (((v25 - 1 - (v25 << 27)) >> 31) ^ (v25 - 1 - ((_DWORD)v25 << 27))); ; m = v19 & v31 )
      {
        v27 = &v18->m128i_i64[3 * m];
        v28 = *v27;
        if ( v17 == *v27 && v27[1] == v20 )
          break;
        if ( v28 == -8 )
        {
          if ( v27[1] == -8 )
          {
            if ( v21 )
              v27 = v21;
            break;
          }
        }
        else if ( v28 == -16 && v27[1] == -16 && !v21 )
        {
          v21 = &v18->m128i_i64[3 * m];
        }
        v31 = v23 + m;
        ++v23;
      }
      *v27 = v17;
      v27[1] = k[1];
      v27[2] = k[2];
      a1->m128i_i32[2] = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
    }
  }
  j___libc_free_0(v4);
}
