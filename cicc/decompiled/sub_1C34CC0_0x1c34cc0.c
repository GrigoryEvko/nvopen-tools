// Function: sub_1C34CC0
// Address: 0x1c34cc0
//
const char *__fastcall sub_1C34CC0(_DWORD *a1, __int64 a2)
{
  const char *result; // rax
  __int64 v4; // rdx
  unsigned __int8 v5; // al
  unsigned __int8 v6; // dl
  __int64 *v7; // rax
  __m128i *v8; // rdx
  __m128i v9; // xmm0
  __int64 *v10; // rax
  __m128i *v11; // rdx
  __m128i v12; // xmm0
  const char *v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // rax
  __m128i *v16; // rdx
  __m128i v17; // xmm0
  size_t v18; // rdx
  char *v19; // rbx
  _BYTE *v20; // r13
  __int64 v21; // rax
  char *v22; // rdi
  char v23; // cl
  __int64 *v24; // rax
  __m128i *v25; // rdx
  __int64 v26; // r15
  __m128i si128; // xmm0
  __m128i *v28; // rdi
  unsigned __int64 v29; // rax
  __m128i v30; // xmm0
  _QWORD *v31; // rdi
  _QWORD *v32; // rdx
  __int64 v33; // rax
  __int64 v34; // r15
  __int64 *v35; // rax
  __m128i *v36; // rdx
  __int64 v37; // r15
  __m128i v38; // xmm0
  __m128i *v39; // rdi
  unsigned __int64 v40; // rax
  __m128i v41; // xmm0
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  _QWORD *v45; // rdi
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // [rsp+0h] [rbp-A0h]
  _BYTE *v50; // [rsp+28h] [rbp-78h] BYREF
  _QWORD *v51; // [rsp+30h] [rbp-70h] BYREF
  _QWORD *v52; // [rsp+38h] [rbp-68h]
  _QWORD dest[2]; // [rsp+40h] [rbp-60h] BYREF
  size_t v54; // [rsp+50h] [rbp-50h] BYREF
  __int64 v55; // [rsp+58h] [rbp-48h]
  _QWORD v56[8]; // [rsp+60h] [rbp-40h] BYREF

  result = sub_15E64D0(a2);
  if ( v4 != 13
    || *(_QWORD *)result != 0x74656D2E6D766C6CLL
    || *((_DWORD *)result + 2) != 1952539745
    || result[12] != 97 )
  {
    if ( (*(_BYTE *)(a2 + 23) & 0x20) == 0 )
    {
LABEL_3:
      v5 = *(_BYTE *)(a2 + 32);
      v6 = v5 & 0xF;
      goto LABEL_4;
    }
    v51 = dest;
    v19 = (char *)sub_1649960(a2);
    v20 = (_BYTE *)v18;
    if ( !v19 )
    {
      LOBYTE(dest[0]) = 0;
      v52 = 0;
      if ( sub_22416F0(&v51, "llvm.", 0, 5) )
      {
        v55 = 0;
        v54 = (size_t)v56;
        LOBYTE(v56[0]) = 0;
        v21 = sub_22416F0(&v54, "nvvm.", 0, 5);
LABEL_30:
        v49 = v21;
        if ( (_QWORD *)v54 != v56 )
          j_j___libc_free_0(v54, v56[0] + 1LL);
        if ( v51 != dest )
          j_j___libc_free_0(v51, dest[0] + 1LL);
        if ( !v49 )
          goto LABEL_3;
        if ( !*a1 )
        {
          v5 = *(_BYTE *)(a2 + 32);
          v6 = v5 & 0xF;
          if ( v19 != &v20[(_QWORD)v19] )
          {
            v22 = v19;
            while ( 1 )
            {
              v23 = *v22;
              if ( (unsigned __int8)((*v22 & 0xDF) - 65) > 0x19u
                && v23 != 36
                && v23 != 95
                && (v19 == v22 || (unsigned __int8)(v23 - 48) > 9u)
                && ((unsigned int)v6 - 7 > 1 || (unsigned __int8)(v23 - 45) > 1u) )
              {
                break;
              }
              if ( ++v22 == &v20[(_QWORD)v19] )
                goto LABEL_4;
            }
            v24 = sub_1C31CC0((__int64)a1, a2, 0);
            v25 = (__m128i *)v24[3];
            v26 = (__int64)v24;
            if ( (unsigned __int64)(v24[2] - (_QWORD)v25) <= 0x18 )
            {
              v43 = sub_16E7EE0((__int64)v24, "Invalid identifier name: ", 0x19u);
              v28 = *(__m128i **)(v43 + 24);
              v26 = v43;
            }
            else
            {
              si128 = _mm_load_si128((const __m128i *)&xmmword_42D0980);
              v25[1].m128i_i8[8] = 32;
              v25[1].m128i_i64[0] = 0x3A656D616E207265LL;
              *v25 = si128;
              v28 = (__m128i *)(v24[3] + 25);
              v24[3] = (__int64)v28;
            }
            v29 = *(_QWORD *)(v26 + 16) - (_QWORD)v28;
            if ( (unsigned __int64)v20 > v29 )
            {
              v42 = sub_16E7EE0(v26, v19, (size_t)v20);
              v28 = *(__m128i **)(v42 + 24);
              v26 = v42;
              v29 = *(_QWORD *)(v42 + 16) - (_QWORD)v28;
            }
            else if ( v20 )
            {
              memcpy(v28, v19, (size_t)v20);
              v44 = *(_QWORD *)(v26 + 16);
              v28 = (__m128i *)&v20[*(_QWORD *)(v26 + 24)];
              *(_QWORD *)(v26 + 24) = v28;
              v29 = v44 - (_QWORD)v28;
            }
            if ( v29 <= 0x25 )
            {
              sub_16E7EE0(v26, "  Must match [a-zA-Z$_][a-zA-Z$_0-9]*\n", 0x26u);
            }
            else
            {
              v30 = _mm_load_si128((const __m128i *)&xmmword_42D0990);
              v28[2].m128i_i32[0] = 1564028208;
              v28[2].m128i_i16[2] = 2602;
              *v28 = v30;
              v28[1] = _mm_load_si128((const __m128i *)&xmmword_42D09A0);
              *(_QWORD *)(v26 + 24) += 38LL;
            }
LABEL_52:
            sub_1C31880((__int64)a1);
            goto LABEL_3;
          }
LABEL_4:
          if ( v6 == 6 )
          {
            v13 = sub_1649960(a2);
            if ( v14 == 9 && *(_QWORD *)v13 == 0x6573752E6D766C6CLL && v13[8] == 100 )
            {
LABEL_11:
              if ( ((*(_BYTE *)(a2 + 32) >> 4) & 3u) - 1 > 1 )
                return (const char *)sub_1C34B00((__int64)a1, a2);
              goto LABEL_12;
            }
            v15 = sub_1C31CC0((__int64)a1, a2, 0);
            v16 = (__m128i *)v15[3];
            if ( (unsigned __int64)(v15[2] - (_QWORD)v16) <= 0x23 )
            {
              sub_16E7EE0((__int64)v15, "appending linkage is not supported.\n", 0x24u);
            }
            else
            {
              v17 = _mm_load_si128((const __m128i *)&xmmword_42D09C0);
              v16[2].m128i_i32[0] = 170812517;
              *v16 = v17;
              v16[1] = _mm_load_si128((const __m128i *)&xmmword_42D09D0);
              v15[3] += 36;
            }
          }
          else
          {
            if ( v6 != 9 )
            {
              if ( ((v5 >> 4) & 3u) - 1 > 1 )
                return (const char *)sub_1C34B00((__int64)a1, a2);
LABEL_12:
              v10 = sub_1C31CC0((__int64)a1, a2, 2u);
              v11 = (__m128i *)v10[3];
              if ( (unsigned __int64)(v10[2] - (_QWORD)v11) <= 0x2E )
              {
                sub_16E7EE0((__int64)v10, "Hidden/protected visibility flags are ignored.\n", 0x2Fu);
              }
              else
              {
                v12 = _mm_load_si128((const __m128i *)&xmmword_42D0A00);
                qmemcpy(&v11[2], "s are ignored.\n", 15);
                *v11 = v12;
                v11[1] = _mm_load_si128((const __m128i *)&xmmword_42D0A10);
                v10[3] += 47;
              }
              return (const char *)sub_1C34B00((__int64)a1, a2);
            }
            v7 = sub_1C31CC0((__int64)a1, a2, 0);
            v8 = (__m128i *)v7[3];
            if ( (unsigned __int64)(v7[2] - (_QWORD)v8) <= 0x25 )
            {
              sub_16E7EE0((__int64)v7, "extern_weak linkage is not supported.\n", 0x26u);
            }
            else
            {
              v9 = _mm_load_si128((const __m128i *)&xmmword_42D09E0);
              v8[2].m128i_i32[0] = 1684370546;
              v8[2].m128i_i16[2] = 2606;
              *v8 = v9;
              v8[1] = _mm_load_si128((const __m128i *)&xmmword_42D09F0);
              v7[3] += 38;
            }
          }
          sub_1C31880((__int64)a1);
          goto LABEL_11;
        }
        v54 = (size_t)v56;
        if ( !v19 )
        {
          v55 = 0;
          LOBYTE(v56[0]) = 0;
          goto LABEL_74;
        }
        v51 = v20;
        if ( (unsigned __int64)v20 > 0xF )
        {
          v54 = sub_22409D0(&v54, &v51, 0);
          v45 = (_QWORD *)v54;
          v56[0] = v51;
        }
        else
        {
          if ( v20 == (_BYTE *)1 )
          {
            v32 = v56;
            LOBYTE(v56[0]) = *v19;
            v33 = 1;
LABEL_73:
            v55 = v33;
            *((_BYTE *)v32 + v33) = 0;
LABEL_74:
            v34 = sub_22417D0(&v54, 0, 0);
            if ( (_QWORD *)v54 != v56 )
              j_j___libc_free_0(v54, v56[0] + 1LL);
            if ( v34 == -1 )
              goto LABEL_3;
            v35 = sub_1C31CC0((__int64)a1, a2, 0);
            v36 = (__m128i *)v35[3];
            v37 = (__int64)v35;
            if ( (unsigned __int64)(v35[2] - (_QWORD)v36) <= 0x18 )
            {
              v47 = sub_16E7EE0((__int64)v35, "Invalid identifier name: ", 0x19u);
              v39 = *(__m128i **)(v47 + 24);
              v37 = v47;
            }
            else
            {
              v38 = _mm_load_si128((const __m128i *)&xmmword_42D0980);
              v36[1].m128i_i8[8] = 32;
              v36[1].m128i_i64[0] = 0x3A656D616E207265LL;
              *v36 = v38;
              v39 = (__m128i *)(v35[3] + 25);
              v35[3] = (__int64)v39;
            }
            v40 = *(_QWORD *)(v37 + 16) - (_QWORD)v39;
            if ( (unsigned __int64)v20 > v40 )
            {
              v46 = sub_16E7EE0(v37, v19, (size_t)v20);
              v39 = *(__m128i **)(v46 + 24);
              v37 = v46;
              v40 = *(_QWORD *)(v46 + 16) - (_QWORD)v39;
            }
            else if ( v20 )
            {
              memcpy(v39, v19, (size_t)v20);
              v48 = *(_QWORD *)(v37 + 16);
              v39 = (__m128i *)&v20[*(_QWORD *)(v37 + 24)];
              *(_QWORD *)(v37 + 24) = v39;
              v40 = v48 - (_QWORD)v39;
            }
            if ( v40 <= 0x1E )
            {
              sub_16E7EE0(v37, "  may not have null character.\n", 0x1Fu);
            }
            else
            {
              v41 = _mm_load_si128((const __m128i *)&xmmword_42D09B0);
              qmemcpy(&v39[1], "ull character.\n", 15);
              *v39 = v41;
              *(_QWORD *)(v37 + 24) += 31LL;
            }
            goto LABEL_52;
          }
          if ( !v20 )
          {
            v32 = v56;
            v33 = 0;
            goto LABEL_73;
          }
          v45 = v56;
        }
        memcpy(v45, v19, (size_t)v20);
        v33 = (__int64)v51;
        v32 = (_QWORD *)v54;
        goto LABEL_73;
      }
      goto LABEL_54;
    }
    v54 = v18;
    if ( v18 > 0xF )
    {
      v51 = (_QWORD *)sub_22409D0(&v51, &v54, 0);
      dest[0] = v54;
      memcpy(v51, v19, (size_t)v20);
      v52 = (_QWORD *)v54;
      *((_BYTE *)v51 + v54) = 0;
      if ( sub_22416F0(&v51, "llvm.", 0, 5) )
      {
        v50 = v20;
        v54 = (size_t)v56;
        v54 = sub_22409D0(&v54, &v50, 0);
        v31 = (_QWORD *)v54;
        v56[0] = v50;
        goto LABEL_68;
      }
LABEL_54:
      if ( v51 == dest )
        goto LABEL_3;
      j_j___libc_free_0(v51, dest[0] + 1LL);
      v5 = *(_BYTE *)(a2 + 32);
      v6 = v5 & 0xF;
      goto LABEL_4;
    }
    if ( v18 == 1 )
    {
      LOBYTE(dest[0]) = *v19;
    }
    else if ( v18 )
    {
      memcpy(dest, v19, v18);
      v52 = (_QWORD *)v54;
      *((_BYTE *)v51 + v54) = 0;
LABEL_26:
      if ( sub_22416F0(&v51, "llvm.", 0, 5) )
      {
        v50 = v20;
        v54 = (size_t)v56;
        if ( v20 == (_BYTE *)1 )
        {
          LOBYTE(v56[0]) = *v19;
LABEL_29:
          v55 = (__int64)v50;
          v50[v54] = 0;
          v21 = sub_22416F0(&v54, "nvvm.", 0, 5);
          goto LABEL_30;
        }
        if ( !v20 )
          goto LABEL_29;
        v31 = v56;
LABEL_68:
        memcpy(v31, v19, (size_t)v20);
        goto LABEL_29;
      }
      goto LABEL_54;
    }
    v52 = (_QWORD *)v18;
    *((_BYTE *)dest + v18) = 0;
    goto LABEL_26;
  }
  return result;
}
