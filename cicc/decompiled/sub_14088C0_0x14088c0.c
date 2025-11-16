// Function: sub_14088C0
// Address: 0x14088c0
//
__int64 __fastcall sub_14088C0(const __m128i *a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // dl
  unsigned __int64 v5; // rax
  int v6; // r14d
  __int64 v7; // r15
  const __m128i *v8; // rax
  __m128i *v9; // r12
  bool v10; // zf
  __m128i *k; // rdx
  __int64 m128i_i64; // rcx
  __int64 v13; // rdi
  unsigned __int64 v14; // rcx
  const __m128i *v15; // r8
  int v16; // esi
  __int64 v17; // rdi
  int v18; // r11d
  __int64 v19; // r9
  const __m128i *v20; // r10
  unsigned __int64 v21; // r9
  unsigned __int64 v22; // r9
  int i; // eax
  __m128i *v24; // r9
  __int64 v25; // r14
  int v26; // eax
  __m128i *v27; // r12
  __int64 v28; // r13
  __m128i *v29; // r8
  __m128i *v30; // rax
  __int64 v31; // rdx
  __m128i *j; // rdx
  unsigned __int64 v33; // rcx
  const __m128i *v34; // r9
  int v35; // esi
  __int64 v36; // rdi
  const __m128i *v37; // r14
  __int64 v38; // r11
  int v39; // r13d
  unsigned __int64 v40; // r11
  unsigned __int64 v41; // r11
  int m; // eax
  __m128i *v43; // r11
  __int64 v44; // r15
  __int32 v45; // esi
  __int64 v46; // rax
  int v47; // eax
  __int32 v48; // esi
  __int64 v49; // rax
  _BYTE v50[112]; // [rsp+10h] [rbp-70h] BYREF

  result = a1->m128i_u8[8];
  v4 = a1->m128i_i8[8] & 1;
  if ( a2 <= 3 )
  {
    if ( v4 )
      return result;
    v27 = (__m128i *)a1[1].m128i_i64[0];
    v28 = a1[1].m128i_u32[2];
    a1->m128i_i8[8] = result | 1;
  }
  else
  {
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
      v7 = 16LL * (unsigned int)v5;
      if ( v4 )
        goto LABEL_5;
      v27 = (__m128i *)a1[1].m128i_i64[0];
      v28 = a1[1].m128i_u32[2];
    }
    else
    {
      if ( v4 )
      {
        v7 = 1024;
        v6 = 64;
LABEL_5:
        v8 = a1 + 1;
        v9 = (__m128i *)v50;
        while ( 1 )
        {
          if ( v8->m128i_i64[0] == -2 )
          {
            if ( v8->m128i_i64[1] != -8 )
              goto LABEL_8;
          }
          else if ( v8->m128i_i64[0] != -16 || v8->m128i_i64[1] != -16 )
          {
LABEL_8:
            if ( v9 )
              *v9 = _mm_loadu_si128(v8);
            ++v9;
          }
          if ( ++v8 == &a1[5] )
          {
            a1->m128i_i8[8] &= ~1u;
            result = sub_22077B0(v7);
            v10 = (a1->m128i_i64[1] & 1) == 0;
            a1->m128i_i64[1] &= 1uLL;
            k = (__m128i *)v50;
            a1[1].m128i_i64[0] = result;
            m128i_i64 = result;
            a1[1].m128i_i32[2] = v6;
            if ( !v10 )
            {
              m128i_i64 = (__int64)a1[1].m128i_i64;
              result = (__int64)a1[1].m128i_i64;
              v7 = 64;
            }
            v13 = result + v7;
            while ( 1 )
            {
              if ( m128i_i64 )
              {
                *(_QWORD *)result = -2;
                *(_QWORD *)(result + 8) = -8;
              }
              result += 16;
              if ( v13 == result )
                break;
              m128i_i64 = result;
            }
            while ( 2 )
            {
              if ( v9 == k )
                return result;
              v14 = k->m128i_i64[0];
              if ( k->m128i_i64[0] == -2 )
              {
                if ( k->m128i_i64[1] == -8 )
                {
LABEL_73:
                  ++k;
                  continue;
                }
              }
              else if ( v14 == -16 && k->m128i_i64[1] == -16 )
              {
                goto LABEL_73;
              }
              break;
            }
            if ( (a1->m128i_i8[8] & 1) != 0 )
            {
              v15 = a1 + 1;
              v16 = 3;
            }
            else
            {
              v48 = a1[1].m128i_i32[2];
              v15 = (const __m128i *)a1[1].m128i_i64[0];
              if ( !v48 )
                goto LABEL_89;
              v16 = v48 - 1;
            }
            v17 = k->m128i_i64[1];
            v18 = 1;
            v19 = ((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4);
            v20 = 0;
            v21 = (((v19 | ((unsigned __int64)((unsigned int)v14 ^ (unsigned int)(v14 >> 9)) << 32)) - 1 - (v19 << 32)) >> 22)
                ^ ((v19 | ((unsigned __int64)((unsigned int)v14 ^ (unsigned int)(v14 >> 9)) << 32)) - 1 - (v19 << 32));
            v22 = ((9 * (((v21 - 1 - (v21 << 13)) >> 8) ^ (v21 - 1 - (v21 << 13)))) >> 15)
                ^ (9 * (((v21 - 1 - (v21 << 13)) >> 8) ^ (v21 - 1 - (v21 << 13))));
            for ( i = v16 & (((v22 - 1 - (v22 << 27)) >> 31) ^ (v22 - 1 - ((_DWORD)v22 << 27))); ; i = v16 & v26 )
            {
              v24 = (__m128i *)&v15[i];
              v25 = v24->m128i_i64[0];
              if ( v14 == v24->m128i_i64[0] && v24->m128i_i64[1] == v17 )
                break;
              if ( v25 == -2 )
              {
                if ( v24->m128i_i64[1] == -8 )
                {
                  if ( v20 )
                    v24 = (__m128i *)v20;
                  break;
                }
              }
              else if ( v25 == -16 && v24->m128i_i64[1] == -16 && !v20 )
              {
                v20 = &v15[i];
              }
              v26 = v18 + i;
              ++v18;
            }
            v49 = k->m128i_i64[0];
            v24->m128i_i64[1] = v17;
            v24->m128i_i64[0] = v49;
            result = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
            a1->m128i_i32[2] = result;
            goto LABEL_73;
          }
        }
      }
      v27 = (__m128i *)a1[1].m128i_i64[0];
      v28 = a1[1].m128i_u32[2];
      v7 = 1024;
      v6 = 64;
    }
    v46 = sub_22077B0(v7);
    a1[1].m128i_i32[2] = v6;
    a1[1].m128i_i64[0] = v46;
  }
  v10 = (a1->m128i_i64[1] & 1) == 0;
  a1->m128i_i64[1] &= 1uLL;
  v29 = &v27[v28];
  if ( v10 )
  {
    v30 = (__m128i *)a1[1].m128i_i64[0];
    v31 = a1[1].m128i_u32[2];
  }
  else
  {
    v30 = (__m128i *)&a1[1];
    v31 = 4;
  }
  for ( j = &v30[v31]; j != v30; ++v30 )
  {
    if ( v30 )
    {
      v30->m128i_i64[0] = -2;
      v30->m128i_i64[1] = -8;
    }
  }
  for ( k = v27; v29 != k; ++k )
  {
    v33 = k->m128i_i64[0];
    if ( k->m128i_i64[0] == -2 )
    {
      if ( k->m128i_i64[1] != -8 )
        goto LABEL_44;
    }
    else if ( v33 != -16 || k->m128i_i64[1] != -16 )
    {
LABEL_44:
      if ( (a1->m128i_i8[8] & 1) != 0 )
      {
        v34 = a1 + 1;
        v35 = 3;
      }
      else
      {
        v45 = a1[1].m128i_i32[2];
        v34 = (const __m128i *)a1[1].m128i_i64[0];
        if ( !v45 )
        {
LABEL_89:
          MEMORY[0] = k->m128i_i64[0];
          BUG();
        }
        v35 = v45 - 1;
      }
      v36 = k->m128i_i64[1];
      v37 = 0;
      v38 = ((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4);
      v39 = 1;
      v40 = (((v38 | ((unsigned __int64)((unsigned int)v33 ^ (unsigned int)(v33 >> 9)) << 32)) - 1 - (v38 << 32)) >> 22)
          ^ ((v38 | ((unsigned __int64)((unsigned int)v33 ^ (unsigned int)(v33 >> 9)) << 32)) - 1 - (v38 << 32));
      v41 = ((9 * (((v40 - 1 - (v40 << 13)) >> 8) ^ (v40 - 1 - (v40 << 13)))) >> 15)
          ^ (9 * (((v40 - 1 - (v40 << 13)) >> 8) ^ (v40 - 1 - (v40 << 13))));
      for ( m = v35 & (((v41 - 1 - (v41 << 27)) >> 31) ^ (v41 - 1 - ((_DWORD)v41 << 27))); ; m = v35 & v47 )
      {
        v43 = (__m128i *)&v34[m];
        v44 = v43->m128i_i64[0];
        if ( v33 == v43->m128i_i64[0] && v43->m128i_i64[1] == v36 )
          break;
        if ( v44 == -2 )
        {
          if ( v43->m128i_i64[1] == -8 )
          {
            if ( v37 )
              v43 = (__m128i *)v37;
            break;
          }
        }
        else if ( v44 == -16 && v43->m128i_i64[1] == -16 && !v37 )
        {
          v37 = &v34[m];
        }
        v47 = v39 + m;
        ++v39;
      }
      v43->m128i_i64[0] = k->m128i_i64[0];
      v43->m128i_i64[1] = k->m128i_i64[1];
      a1->m128i_i32[2] = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
    }
  }
  return j___libc_free_0(v27);
}
