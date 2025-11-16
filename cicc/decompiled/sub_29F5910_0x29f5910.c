// Function: sub_29F5910
// Address: 0x29f5910
//
void __fastcall sub_29F5910(__int64 a1, char *a2)
{
  const __m128i *v2; // r15
  int v4; // eax
  __m128i *v5; // rcx
  const void *v6; // r11
  const __m128i *v7; // r8
  size_t v8; // rdx
  signed __int64 v9; // rax
  signed __int64 v10; // rax
  unsigned __int64 v11; // rax
  char *v12; // rbx
  size_t v13; // r12
  size_t v14; // rbx
  __int64 v15; // r14
  __int64 *v16; // r12
  __m128i *v17; // rdi
  unsigned __int64 v18; // rbx
  __int64 v19; // rdx
  __int64 v20; // rax
  _BYTE *v21; // rax
  __int64 *v22; // r14
  size_t v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __m128i *v27; // rdi
  __int64 v28; // rcx
  __int64 v29; // rdx
  _BYTE *v30; // rdi
  size_t n; // [rsp+10h] [rbp-80h]
  void *s1; // [rsp+18h] [rbp-78h]
  const __m128i *s1a; // [rsp+18h] [rbp-78h]
  __m128i *s2; // [rsp+20h] [rbp-70h]
  __m128i *s2a; // [rsp+20h] [rbp-70h]
  __m128i *v36; // [rsp+30h] [rbp-60h]
  size_t v37; // [rsp+38h] [rbp-58h]
  __m128i v38; // [rsp+40h] [rbp-50h] BYREF
  unsigned __int64 v39; // [rsp+50h] [rbp-40h]

  if ( (char *)a1 != a2 && a2 != (char *)(a1 + 40) )
  {
    v2 = (const __m128i *)(a1 + 56);
    while ( 1 )
    {
      v13 = v2[-1].m128i_u64[1];
      v14 = *(_QWORD *)(a1 + 8);
      v7 = v2;
      v5 = (__m128i *)v2[-1].m128i_i64[0];
      v6 = *(const void **)a1;
      v8 = v14;
      if ( v13 <= v14 )
        v8 = v2[-1].m128i_u64[1];
      if ( v8 )
      {
        n = v8;
        s1 = *(void **)a1;
        s2 = (__m128i *)v2[-1].m128i_i64[0];
        v4 = memcmp(s2, *(const void **)a1, v8);
        v5 = s2;
        v6 = s1;
        v7 = v2;
        v8 = n;
        if ( v4 )
        {
          if ( v4 < 0 )
            goto LABEL_22;
LABEL_9:
          s1a = v7;
          s2a = v5;
          LODWORD(v10) = memcmp(v6, v5, v8);
          v5 = s2a;
          v7 = s1a;
          if ( (_DWORD)v10 )
            goto LABEL_12;
          goto LABEL_10;
        }
        v9 = v13 - v14;
        if ( (__int64)(v13 - v14) >= 0x80000000LL )
          goto LABEL_9;
      }
      else
      {
        v9 = v13 - v14;
        if ( (__int64)(v13 - v14) >= 0x80000000LL )
          goto LABEL_10;
      }
      if ( v9 <= (__int64)0xFFFFFFFF7FFFFFFFLL || (int)v9 < 0 )
      {
LABEL_22:
        v11 = v2[1].m128i_u64[0];
        goto LABEL_23;
      }
      if ( v8 )
        goto LABEL_9;
LABEL_10:
      v10 = v14 - v13;
      if ( (__int64)(v14 - v13) >= 0x80000000LL )
        goto LABEL_13;
      if ( v10 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
LABEL_14:
        v12 = &v2[1].m128i_i8[8];
        sub_29F55B0((__m128i *)&v2[-1]);
LABEL_15:
        v2 = (const __m128i *)((char *)v2 + 40);
        if ( a2 == v12 )
          return;
      }
      else
      {
LABEL_12:
        if ( (int)v10 < 0 )
          goto LABEL_14;
LABEL_13:
        v11 = v2[1].m128i_u64[0];
        if ( v11 >= *(_QWORD *)(a1 + 32) )
          goto LABEL_14;
LABEL_23:
        v36 = &v38;
        if ( v5 == v2 )
        {
          v38 = _mm_loadu_si128(v2);
        }
        else
        {
          v36 = v5;
          v38.m128i_i64[0] = v2->m128i_i64[0];
        }
        v15 = (__int64)v2[-1].m128i_i64 - a1;
        v39 = v11;
        v12 = &v2[1].m128i_i8[8];
        v37 = v13;
        v2[-1].m128i_i64[0] = (__int64)v2;
        v2[-1].m128i_i64[1] = 0;
        v2->m128i_i8[0] = 0;
        if ( v15 > 0 )
        {
          v16 = &v2[-3].m128i_i64[1];
          v17 = (__m128i *)v7;
          v18 = 0xCCCCCCCCCCCCCCCDLL * (v15 >> 3);
          while ( 1 )
          {
            v22 = (__int64 *)*(v16 - 2);
            if ( v16 == v22 )
            {
              v23 = *(v16 - 1);
              if ( v23 )
              {
                if ( v23 == 1 )
                  v17->m128i_i8[0] = *(_BYTE *)v16;
                else
                  memcpy(v17, v16, v23);
              }
              v24 = *(v22 - 1);
              v25 = v22[3];
              v22[4] = v24;
              *(_BYTE *)(v25 + v24) = 0;
            }
            else
            {
              if ( v17 == (__m128i *)(v16 + 5) )
              {
                v26 = *(v16 - 1);
                v16[3] = (__int64)v22;
                v16[4] = v26;
                v16[5] = *v16;
              }
              else
              {
                v19 = *(v16 - 1);
                v20 = v16[5];
                v16[3] = (__int64)v22;
                v16[4] = v19;
                v16[5] = *v16;
                if ( v17 )
                {
                  *(v16 - 2) = (__int64)v17;
                  *v16 = v20;
                  goto LABEL_30;
                }
              }
              *(v16 - 2) = (__int64)v16;
            }
LABEL_30:
            v21 = (_BYTE *)*(v16 - 2);
            v16 -= 5;
            v16[4] = 0;
            *v21 = 0;
            v16[12] = v16[7];
            if ( !--v18 )
            {
              v12 = &v2[1].m128i_i8[8];
              v13 = v37;
              break;
            }
            v17 = (__m128i *)v16[3];
          }
        }
        v27 = *(__m128i **)a1;
        if ( v36 == &v38 )
        {
          if ( !v13 )
            goto LABEL_52;
          if ( v13 != 1 )
          {
            memcpy(v27, &v38, v13);
            v13 = v37;
            v27 = *(__m128i **)a1;
LABEL_52:
            *(_QWORD *)(a1 + 8) = v13;
            v27->m128i_i8[v13] = 0;
            v27 = v36;
            goto LABEL_44;
          }
          v27->m128i_i8[0] = v38.m128i_i8[0];
          v30 = *(_BYTE **)a1;
          *(_QWORD *)(a1 + 8) = v37;
          v30[v37] = 0;
          v27 = v36;
        }
        else
        {
          v28 = v38.m128i_i64[0];
          if ( v27 == (__m128i *)(a1 + 16) )
          {
            *(_QWORD *)a1 = v36;
            *(_QWORD *)(a1 + 8) = v13;
            *(_QWORD *)(a1 + 16) = v28;
          }
          else
          {
            v29 = *(_QWORD *)(a1 + 16);
            *(_QWORD *)a1 = v36;
            *(_QWORD *)(a1 + 8) = v13;
            *(_QWORD *)(a1 + 16) = v28;
            if ( v27 )
            {
              v36 = v27;
              v38.m128i_i64[0] = v29;
              goto LABEL_44;
            }
          }
          v36 = &v38;
          v27 = &v38;
        }
LABEL_44:
        v27->m128i_i8[0] = 0;
        *(_QWORD *)(a1 + 32) = v39;
        if ( v36 == &v38 )
          goto LABEL_15;
        v2 = (const __m128i *)((char *)v2 + 40);
        j_j___libc_free_0((unsigned __int64)v36);
        if ( a2 == v12 )
          return;
      }
    }
  }
}
