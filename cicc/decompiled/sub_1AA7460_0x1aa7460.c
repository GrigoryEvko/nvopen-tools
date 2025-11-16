// Function: sub_1AA7460
// Address: 0x1aa7460
//
__int64 __fastcall sub_1AA7460(const __m128i *a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // dl
  __int64 *v5; // r12
  unsigned __int64 v6; // rax
  int v7; // r14d
  __int64 v8; // r13
  __int64 v9; // r15
  __int64 v10; // rax
  bool v11; // zf
  __int64 *v12; // r8
  __m128i *v13; // rax
  __int64 v14; // rdx
  const __m128i *v15; // rax
  __m128i *v16; // r12
  __m128i *v17; // rdx
  __int64 m128i_i64; // rcx
  __int64 v19; // rdi
  __int64 v20; // rcx
  const __m128i *v21; // r8
  int v22; // esi
  __int64 v23; // rdi
  int v24; // r11d
  __int64 v25; // r9
  const __m128i *v26; // r10
  unsigned __int64 v27; // r9
  unsigned __int64 v28; // r9
  int i; // eax
  __m128i *v30; // r9
  __int64 v31; // r14
  int v32; // eax
  __int32 v33; // esi
  __int64 v34; // r13
  __m128i *j; // rdx
  __int64 *k; // rdx
  const __m128i *v37; // r9
  int v38; // esi
  __int64 v39; // rdi
  const __m128i *v40; // r14
  __int64 v41; // r11
  int v42; // r13d
  unsigned __int64 v43; // r11
  unsigned __int64 v44; // r11
  int m; // eax
  __m128i *v46; // r11
  __int64 v47; // r15
  __int32 v48; // esi
  int v49; // eax
  _BYTE v50[80]; // [rsp+10h] [rbp-50h] BYREF

  result = a1->m128i_u8[8];
  v4 = a1->m128i_i8[8] & 1;
  if ( a2 <= 1 )
  {
    if ( v4 )
      return result;
    v34 = a1[1].m128i_u32[2];
    v5 = (__int64 *)a1[1].m128i_i64[0];
    a1->m128i_i8[8] = result | 1;
    v11 = (a1->m128i_i64[1] & 1) == 0;
    a1->m128i_i64[1] &= 1uLL;
    v12 = &v5[2 * v34];
    if ( v11 )
      goto LABEL_6;
LABEL_48:
    v13 = (__m128i *)&a1[1];
    v14 = 2;
    goto LABEL_49;
  }
  v5 = (__int64 *)a1[1].m128i_i64[0];
  v6 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
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
  v7 = v6;
  if ( (unsigned int)v6 <= 0x40 )
  {
    if ( !v4 )
    {
      v8 = a1[1].m128i_u32[2];
      v9 = 1024;
      v7 = 64;
      goto LABEL_5;
    }
    v9 = 1024;
    v7 = 64;
    goto LABEL_10;
  }
  v9 = 16LL * (unsigned int)v6;
  if ( v4 )
  {
LABEL_10:
    v15 = a1 + 1;
    v16 = (__m128i *)v50;
    while ( 1 )
    {
      if ( v15->m128i_i64[0] == -8 )
      {
        if ( v15->m128i_i64[1] != -8 )
          goto LABEL_13;
      }
      else if ( v15->m128i_i64[0] != -16 || v15->m128i_i64[1] != -16 )
      {
LABEL_13:
        if ( v16 )
          *v16 = _mm_loadu_si128(v15);
        ++v16;
      }
      if ( ++v15 == &a1[3] )
      {
        a1->m128i_i8[8] &= ~1u;
        result = sub_22077B0(v9);
        v11 = (a1->m128i_i64[1] & 1) == 0;
        a1->m128i_i64[1] &= 1uLL;
        v17 = (__m128i *)v50;
        a1[1].m128i_i64[0] = result;
        m128i_i64 = result;
        a1[1].m128i_i32[2] = v7;
        if ( !v11 )
        {
          m128i_i64 = (__int64)a1[1].m128i_i64;
          result = (__int64)a1[1].m128i_i64;
          v9 = 32;
        }
        v19 = result + v9;
        while ( 1 )
        {
          if ( m128i_i64 )
          {
            *(_QWORD *)result = -8;
            *(_QWORD *)(result + 8) = -8;
          }
          result += 16;
          if ( v19 == result )
            break;
          m128i_i64 = result;
        }
        if ( v16 == (__m128i *)v50 )
          return result;
        while ( 2 )
        {
          v20 = v17->m128i_i64[0];
          if ( v17->m128i_i64[0] == -8 )
          {
            if ( v17->m128i_i64[1] == -8 )
              goto LABEL_40;
          }
          else if ( v20 == -16 && v17->m128i_i64[1] == -16 )
          {
            goto LABEL_40;
          }
          if ( (a1->m128i_i8[8] & 1) != 0 )
          {
            v21 = a1 + 1;
            v22 = 1;
          }
          else
          {
            v33 = a1[1].m128i_i32[2];
            v21 = (const __m128i *)a1[1].m128i_i64[0];
            if ( !v33 )
            {
LABEL_88:
              MEMORY[0] = v20;
              BUG();
            }
            v22 = v33 - 1;
          }
          v23 = v17->m128i_i64[1];
          v24 = 1;
          v25 = ((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4);
          v26 = 0;
          v27 = (((v25 | ((unsigned __int64)(((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4)) << 32))
                - 1
                - (v25 << 32)) >> 22)
              ^ ((v25 | ((unsigned __int64)(((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4)) << 32))
               - 1
               - (v25 << 32));
          v28 = ((9 * (((v27 - 1 - (v27 << 13)) >> 8) ^ (v27 - 1 - (v27 << 13)))) >> 15)
              ^ (9 * (((v27 - 1 - (v27 << 13)) >> 8) ^ (v27 - 1 - (v27 << 13))));
          for ( i = v22 & (((v28 - 1 - (v28 << 27)) >> 31) ^ (v28 - 1 - ((_DWORD)v28 << 27))); ; i = v22 & v32 )
          {
            v30 = (__m128i *)&v21[i];
            v31 = v30->m128i_i64[0];
            if ( v20 == v30->m128i_i64[0] && v30->m128i_i64[1] == v23 )
              break;
            if ( v31 == -8 )
            {
              if ( v30->m128i_i64[1] == -8 )
              {
                if ( v26 )
                  v30 = (__m128i *)v26;
                break;
              }
            }
            else if ( v31 == -16 && v30->m128i_i64[1] == -16 && !v26 )
            {
              v26 = &v21[i];
            }
            v32 = v24 + i;
            ++v24;
          }
          v30->m128i_i64[0] = v20;
          v30->m128i_i64[1] = v23;
          result = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
          a1->m128i_i32[2] = result;
LABEL_40:
          if ( v16 == ++v17 )
            return result;
          continue;
        }
      }
    }
  }
  v8 = a1[1].m128i_u32[2];
LABEL_5:
  v10 = sub_22077B0(v9);
  v11 = (a1->m128i_i64[1] & 1) == 0;
  a1->m128i_i64[1] &= 1uLL;
  v12 = &v5[2 * v8];
  a1[1].m128i_i64[0] = v10;
  a1[1].m128i_i32[2] = v7;
  if ( !v11 )
    goto LABEL_48;
LABEL_6:
  v13 = (__m128i *)a1[1].m128i_i64[0];
  v14 = a1[1].m128i_u32[2];
LABEL_49:
  for ( j = &v13[v14]; j != v13; ++v13 )
  {
    if ( v13 )
    {
      v13->m128i_i64[0] = -8;
      v13->m128i_i64[1] = -8;
    }
  }
  for ( k = v5; v12 != k; k += 2 )
  {
    v20 = *k;
    if ( *k == -8 )
    {
      if ( k[1] != -8 )
        goto LABEL_56;
    }
    else if ( v20 != -16 || k[1] != -16 )
    {
LABEL_56:
      if ( (a1->m128i_i8[8] & 1) != 0 )
      {
        v37 = a1 + 1;
        v38 = 1;
      }
      else
      {
        v48 = a1[1].m128i_i32[2];
        v37 = (const __m128i *)a1[1].m128i_i64[0];
        if ( !v48 )
          goto LABEL_88;
        v38 = v48 - 1;
      }
      v39 = k[1];
      v40 = 0;
      v41 = ((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4);
      v42 = 1;
      v43 = (((v41 | ((unsigned __int64)(((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4)) << 32)) - 1 - (v41 << 32)) >> 22)
          ^ ((v41 | ((unsigned __int64)(((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4)) << 32)) - 1 - (v41 << 32));
      v44 = ((9 * (((v43 - 1 - (v43 << 13)) >> 8) ^ (v43 - 1 - (v43 << 13)))) >> 15)
          ^ (9 * (((v43 - 1 - (v43 << 13)) >> 8) ^ (v43 - 1 - (v43 << 13))));
      for ( m = v38 & (((v44 - 1 - (v44 << 27)) >> 31) ^ (v44 - 1 - ((_DWORD)v44 << 27))); ; m = v38 & v49 )
      {
        v46 = (__m128i *)&v37[m];
        v47 = v46->m128i_i64[0];
        if ( v20 == v46->m128i_i64[0] && v46->m128i_i64[1] == v39 )
          break;
        if ( v47 == -8 )
        {
          if ( v46->m128i_i64[1] == -8 )
          {
            if ( v40 )
              v46 = (__m128i *)v40;
            break;
          }
        }
        else if ( v47 == -16 && v46->m128i_i64[1] == -16 && !v40 )
        {
          v40 = &v37[m];
        }
        v49 = v42 + m;
        ++v42;
      }
      v46->m128i_i64[0] = v20;
      v46->m128i_i64[1] = k[1];
      a1->m128i_i32[2] = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
    }
  }
  return j___libc_free_0(v5);
}
