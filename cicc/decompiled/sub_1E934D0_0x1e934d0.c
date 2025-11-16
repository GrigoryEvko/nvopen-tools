// Function: sub_1E934D0
// Address: 0x1e934d0
//
void __fastcall sub_1E934D0(__int64 a1, char *a2)
{
  const __m128i *v3; // r15
  size_t v4; // r13
  size_t v5; // rbx
  const __m128i *v6; // r12
  const __m128i *v7; // r8
  __m128i *v8; // rdi
  size_t v9; // rdx
  signed __int64 v10; // rax
  __int64 v11; // rax
  char *v12; // r10
  __int64 v13; // rax
  unsigned __int64 v14; // r12
  __int64 *v15; // rbx
  __m128i *i; // rdi
  __int64 v17; // rdx
  __int64 v18; // rax
  _BYTE *v19; // rax
  __int64 *v20; // r13
  size_t v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __m128i *v25; // rdi
  __int64 v26; // rsi
  __int64 v27; // rdx
  _BYTE *v28; // rdi
  __m128i *v30; // [rsp+10h] [rbp-70h]
  char *v31; // [rsp+10h] [rbp-70h]
  char *v32; // [rsp+10h] [rbp-70h]
  __m128i *v33; // [rsp+20h] [rbp-60h]
  size_t v34; // [rsp+28h] [rbp-58h]
  __m128i v35; // [rsp+30h] [rbp-50h] BYREF
  __int64 v36; // [rsp+40h] [rbp-40h]

  if ( (char *)a1 != a2 && a2 != (char *)(a1 + 40) )
  {
    v3 = (const __m128i *)(a1 + 56);
    do
    {
      v4 = v3[-1].m128i_u64[1];
      v5 = *(_QWORD *)(a1 + 8);
      v6 = v3 - 1;
      v7 = v3;
      v8 = (__m128i *)v3[-1].m128i_i64[0];
      v9 = v5;
      if ( v4 <= v5 )
        v9 = v3[-1].m128i_u64[1];
      if ( !v9
        || (v30 = (__m128i *)v3[-1].m128i_i64[0],
            LODWORD(v10) = memcmp(v8, *(const void **)a1, v9),
            v8 = v30,
            v7 = v3,
            !(_DWORD)v10) )
      {
        v10 = v4 - v5;
        if ( (__int64)(v4 - v5) >= 0x80000000LL )
          goto LABEL_36;
        if ( v10 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
          goto LABEL_11;
      }
      if ( (int)v10 < 0 )
      {
LABEL_11:
        v33 = &v35;
        if ( v8 == v3 )
        {
          v35 = _mm_loadu_si128(v3);
        }
        else
        {
          v33 = v8;
          v35.m128i_i64[0] = v3->m128i_i64[0];
        }
        v11 = v3[1].m128i_i64[0];
        v34 = v4;
        v12 = &v3[1].m128i_i8[8];
        v3[-1].m128i_i64[0] = (__int64)v3;
        v36 = v11;
        v13 = (__int64)v6->m128i_i64 - a1;
        v3[-1].m128i_i64[1] = 0;
        v3->m128i_i8[0] = 0;
        v14 = 0xCCCCCCCCCCCCCCCDLL * (((__int64)v6->m128i_i64 - a1) >> 3);
        if ( v13 > 0 )
        {
          v15 = &v3[-3].m128i_i64[1];
          for ( i = (__m128i *)v7; ; i = (__m128i *)v15[3] )
          {
            v20 = (__int64 *)*(v15 - 2);
            if ( v15 == v20 )
            {
              v21 = *(v15 - 1);
              if ( v21 )
              {
                if ( v21 == 1 )
                  i->m128i_i8[0] = *(_BYTE *)v15;
                else
                  memcpy(i, v15, v21);
              }
              v22 = *(v20 - 1);
              v23 = v20[3];
              v20[4] = v22;
              *(_BYTE *)(v23 + v22) = 0;
            }
            else
            {
              if ( i == (__m128i *)(v15 + 5) )
              {
                v24 = *(v15 - 1);
                v15[3] = (__int64)v20;
                v15[4] = v24;
                v15[5] = *v15;
              }
              else
              {
                v17 = *(v15 - 1);
                v18 = v15[5];
                v15[3] = (__int64)v20;
                v15[4] = v17;
                v15[5] = *v15;
                if ( i )
                {
                  *(v15 - 2) = (__int64)i;
                  *v15 = v18;
                  goto LABEL_18;
                }
              }
              *(v15 - 2) = (__int64)v15;
            }
LABEL_18:
            v19 = (_BYTE *)*(v15 - 2);
            *(v15 - 1) = 0;
            v15 -= 5;
            *v19 = 0;
            v15[12] = v15[7];
            if ( !--v14 )
            {
              v12 = &v3[1].m128i_i8[8];
              v4 = v34;
              break;
            }
          }
        }
        v25 = *(__m128i **)a1;
        if ( v33 != &v35 )
        {
          v26 = v35.m128i_i64[0];
          if ( v25 == (__m128i *)(a1 + 16) )
          {
            *(_QWORD *)a1 = v33;
            *(_QWORD *)(a1 + 8) = v4;
            *(_QWORD *)(a1 + 16) = v26;
          }
          else
          {
            v27 = *(_QWORD *)(a1 + 16);
            *(_QWORD *)a1 = v33;
            *(_QWORD *)(a1 + 8) = v4;
            *(_QWORD *)(a1 + 16) = v26;
            if ( v25 )
            {
              v33 = v25;
              v35.m128i_i64[0] = v27;
              goto LABEL_32;
            }
          }
          v33 = &v35;
          v25 = &v35;
LABEL_32:
          v25->m128i_i8[0] = 0;
          *(_QWORD *)(a1 + 32) = v36;
          if ( v33 != &v35 )
          {
            v31 = v12;
            j_j___libc_free_0(v33, v35.m128i_i64[0] + 1);
            v12 = v31;
          }
          goto LABEL_34;
        }
        if ( v4 )
        {
          if ( v4 == 1 )
          {
            v25->m128i_i8[0] = v35.m128i_i8[0];
            v28 = *(_BYTE **)a1;
            *(_QWORD *)(a1 + 8) = v34;
            v28[v34] = 0;
            v25 = v33;
            goto LABEL_32;
          }
          v32 = v12;
          memcpy(v25, &v35, v4);
          v4 = v34;
          v25 = *(__m128i **)a1;
          v12 = v32;
        }
        *(_QWORD *)(a1 + 8) = v4;
        v25->m128i_i8[v4] = 0;
        v25 = v33;
        goto LABEL_32;
      }
LABEL_36:
      sub_1E93240((__m128i *)&v3[-1]);
      v12 = &v3[1].m128i_i8[8];
LABEL_34:
      v3 = (const __m128i *)((char *)v3 + 40);
    }
    while ( a2 != v12 );
  }
}
