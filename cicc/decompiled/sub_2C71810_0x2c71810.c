// Function: sub_2C71810
// Address: 0x2c71810
//
__int64 __fastcall sub_2C71810(__int64 *a1, __int64 a2)
{
  int v3; // esi
  __m128i *v4; // rdx
  __m128i si128; // xmm0
  __int64 v6; // r12
  const char *v7; // rax
  size_t v8; // rdx
  char *v9; // rdi
  unsigned __int8 *v10; // rsi
  unsigned __int64 v11; // rax
  __int64 *v12; // rdi
  __int64 *v13; // r12
  __int64 *v14; // rbx
  __int64 v15; // rsi
  __int64 *v16; // rax
  _WORD *v17; // rdx
  unsigned int *v18; // r14
  unsigned __int64 v20; // rax
  __int64 v21; // rax
  __int64 *v22; // rax
  __int64 *v23; // r14
  __int64 *v24; // rbx
  __int64 *v25; // r12
  __int64 **v26; // rcx
  __int64 *v27; // r14
  __int64 v28; // rbx
  unsigned __int64 v29; // rax
  __int64 *v30; // r15
  __int64 v31; // r14
  const char *v32; // rax
  size_t v33; // rdx
  size_t v34; // rbx
  const char *v35; // rax
  size_t v36; // rdx
  size_t v37; // rcx
  int v38; // eax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 *i; // [rsp+0h] [rbp-70h]
  size_t v43; // [rsp+10h] [rbp-60h]
  char *s2a; // [rsp+18h] [rbp-58h]
  const char *s2; // [rsp+18h] [rbp-58h]
  __int64 **s2b; // [rsp+18h] [rbp-58h]
  __int64 *v47; // [rsp+20h] [rbp-50h] BYREF
  __int64 *v48; // [rsp+28h] [rbp-48h]
  __int64 *v49; // [rsp+30h] [rbp-40h]

  v3 = *((_DWORD *)a1 + 6);
  v47 = 0;
  v48 = 0;
  v49 = 0;
  if ( v3 )
  {
    v22 = (__int64 *)a1[2];
    v23 = &v22[11 * *((unsigned int *)a1 + 8)];
    if ( v22 != v23 )
    {
      while ( 1 )
      {
        v24 = v22;
        if ( *v22 != -4096 && *v22 != -8192 )
          break;
        v22 += 11;
        if ( v23 == v22 )
          goto LABEL_2;
      }
      if ( v23 != v22 )
      {
        v25 = 0;
        v26 = &v47;
LABEL_51:
        s2b = v26;
        sub_F46430((__int64)v26, v25, v24);
        v25 = v48;
        v26 = s2b;
        while ( 1 )
        {
          v24 += 11;
          if ( v24 == v23 )
            break;
          while ( *v24 == -8192 || *v24 == -4096 )
          {
            v24 += 11;
            if ( v23 == v24 )
              goto LABEL_36;
          }
          if ( v23 == v24 )
            break;
          if ( v25 == v49 )
            goto LABEL_51;
          if ( v25 )
          {
            *v25 = *v24;
            v25 = v48;
          }
          v48 = ++v25;
        }
LABEL_36:
        v27 = v47;
        if ( v47 != v25 )
        {
          v28 = (char *)v25 - (char *)v47;
          _BitScanReverse64(&v29, v25 - v47);
          sub_2C6E6B0(v47, (char *)v25, 2LL * (int)(63 - (v29 ^ 0x3F)));
          if ( v28 <= 128 )
          {
            sub_2C6E550(v27, v25);
          }
          else
          {
            sub_2C6E550(v27, v27 + 16);
            for ( i = v27 + 16; v25 != i; *v30 = v31 )
            {
              v30 = i;
              v31 = *i;
              while ( 1 )
              {
                while ( 1 )
                {
                  v32 = sub_BD5D20(*(v30 - 1));
                  v34 = v33;
                  s2 = v32;
                  v35 = sub_BD5D20(v31);
                  v37 = v36;
                  if ( v34 <= v36 )
                    v36 = v34;
                  if ( !v36 )
                    break;
                  v43 = v37;
                  v38 = memcmp(v35, s2, v36);
                  v37 = v43;
                  if ( !v38 )
                    break;
                  if ( v38 >= 0 )
                    goto LABEL_40;
                  v40 = *--v30;
                  v30[1] = v40;
                }
                if ( v34 == v37 || v34 <= v37 )
                  break;
                v39 = *--v30;
                v30[1] = v39;
              }
LABEL_40:
              ++i;
            }
          }
        }
      }
    }
  }
LABEL_2:
  v4 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v4 <= 0x11u )
  {
    v6 = sub_CB6200(a2, "Merge sets for fn ", 0x12u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_43A1910);
    v6 = a2;
    v4[1].m128i_i16[0] = 8302;
    *v4 = si128;
    *(_QWORD *)(a2 + 32) += 18LL;
  }
  v7 = sub_BD5D20(*(_QWORD *)(**(_QWORD **)(*a1 + 96) + 72LL));
  v9 = *(char **)(v6 + 32);
  v10 = (unsigned __int8 *)v7;
  v11 = *(_QWORD *)(v6 + 24) - (_QWORD)v9;
  if ( v11 < v8 )
  {
    v21 = sub_CB6200(v6, v10, v8);
    v9 = *(char **)(v21 + 32);
    v6 = v21;
    v11 = *(_QWORD *)(v21 + 24) - (_QWORD)v9;
  }
  else if ( v8 )
  {
    s2a = (char *)v8;
    memcpy(v9, v10, v8);
    v9 = &s2a[*(_QWORD *)(v6 + 32)];
    v20 = *(_QWORD *)(v6 + 24) - (_QWORD)v9;
    *(_QWORD *)(v6 + 32) = v9;
    if ( v20 > 1 )
      goto LABEL_7;
    goto LABEL_19;
  }
  if ( v11 > 1 )
  {
LABEL_7:
    *(_WORD *)v9 = 2618;
    *(_QWORD *)(v6 + 32) += 2LL;
    goto LABEL_8;
  }
LABEL_19:
  sub_CB6200(v6, (unsigned __int8 *)":\n", 2u);
LABEL_8:
  v12 = v47;
  v13 = v48;
  if ( v48 != v47 )
  {
    v14 = v47;
    do
    {
      v16 = sub_2C6ECE0((__int64)a1, *v14);
      v17 = *(_WORD **)(a2 + 32);
      v18 = (unsigned int *)v16;
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v17 > 1u )
      {
        v15 = a2;
        *v17 = 8224;
        *(_QWORD *)(a2 + 32) += 2LL;
      }
      else
      {
        v15 = sub_CB6200(a2, (unsigned __int8 *)"  ", 2u);
      }
      ++v14;
      sub_2C71740(v18, v15, a1);
    }
    while ( v13 != v14 );
    v12 = v47;
  }
  if ( v12 )
    j_j___libc_free_0((unsigned __int64)v12);
  return a2;
}
