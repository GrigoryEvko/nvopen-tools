// Function: sub_2C797D0
// Address: 0x2c797d0
//
void __fastcall sub_2C797D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  const char *v4; // rax
  __int64 v5; // rdx
  unsigned __int8 v6; // al
  unsigned __int8 v7; // dl
  const char *v8; // rsi
  __int64 *v9; // rax
  __m128i *v10; // rdx
  __m128i v11; // xmm0
  __int64 v12; // rcx
  const char *v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // rax
  __m128i *v16; // rdx
  __m128i v17; // xmm0
  size_t v18; // rdx
  unsigned __int8 *v19; // rbx
  _BYTE *v20; // r13
  __int64 v21; // rax
  unsigned __int8 *v22; // rdi
  unsigned __int8 v23; // cl
  __int64 *v24; // rax
  char *v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // r15
  __m128i si128; // xmm0
  __m128i *v29; // rdi
  unsigned __int64 v30; // rax
  __m128i v31; // xmm0
  __int64 *v32; // rax
  __m128i *v33; // rdx
  __m128i v34; // xmm0
  _BYTE *v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rdi
  _QWORD *v38; // rdi
  _QWORD *v39; // rdx
  __int64 v40; // rax
  char *v41; // r15
  __int64 *v42; // rax
  __int64 v43; // r15
  __m128i v44; // xmm0
  __m128i *v45; // rdi
  unsigned __int64 v46; // rax
  __m128i v47; // xmm0
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  _QWORD *v51; // rdi
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // [rsp+0h] [rbp-A0h]
  _BYTE *v56; // [rsp+28h] [rbp-78h] BYREF
  _QWORD *v57; // [rsp+30h] [rbp-70h] BYREF
  __int64 v58; // [rsp+38h] [rbp-68h]
  _QWORD dest[2]; // [rsp+40h] [rbp-60h] BYREF
  __int64 v60; // [rsp+50h] [rbp-50h] BYREF
  unsigned __int64 v61; // [rsp+58h] [rbp-48h]
  _QWORD v62[8]; // [rsp+60h] [rbp-40h] BYREF

  v2 = a2;
  v4 = sub_B32650((_BYTE *)a2, a2);
  if ( v5 != 13 || *(_QWORD *)v4 != 0x74656D2E6D766C6CLL || *((_DWORD *)v4 + 2) != 1952539745 || v4[12] != 97 )
  {
    if ( (*(_BYTE *)(a2 + 7) & 0x10) == 0 )
    {
LABEL_4:
      v6 = *(_BYTE *)(v2 + 32);
      v7 = v6 & 0xF;
      goto LABEL_5;
    }
    v57 = dest;
    v19 = (unsigned __int8 *)sub_BD5D20(a2);
    v20 = (_BYTE *)v18;
    if ( !v19 )
    {
      LOBYTE(dest[0]) = 0;
      v58 = 0;
      if ( sub_22416F0((__int64 *)&v57, (char *)"llvm.", 0, 5u) )
      {
        v61 = 0;
        v60 = (__int64)v62;
        LOBYTE(v62[0]) = 0;
        v21 = sub_22416F0(&v60, "nvvm.", 0, 5u);
LABEL_23:
        v55 = v21;
        if ( (_QWORD *)v60 != v62 )
          j_j___libc_free_0(v60);
        if ( v57 != dest )
          j_j___libc_free_0((unsigned __int64)v57);
        if ( !v55 )
          goto LABEL_4;
        if ( !*(_DWORD *)a1 )
        {
          v6 = *(_BYTE *)(a2 + 32);
          v7 = v6 & 0xF;
          if ( &v20[(_QWORD)v19] != v19 )
          {
            v22 = v19;
            while ( 1 )
            {
              v23 = *v22;
              if ( (unsigned __int8)((*v22 & 0xDF) - 65) > 0x19u
                && v23 != 36
                && v23 != 95
                && (v22 == v19 || (unsigned __int8)(v23 - 48) > 9u)
                && ((unsigned int)v7 - 7 > 1 || (unsigned __int8)(v23 - 45) > 1u) )
              {
                break;
              }
              if ( &v20[(_QWORD)v19] == ++v22 )
                goto LABEL_5;
            }
            v24 = sub_2C76620(a1, a2, 0);
            v26 = v24[4];
            v27 = (__int64)v24;
            if ( (unsigned __int64)(v24[3] - v26) <= 0x18 )
            {
              a2 = (__int64)"Invalid identifier name: ";
              v49 = sub_CB6200((__int64)v24, "Invalid identifier name: ", 0x19u);
              v29 = *(__m128i **)(v49 + 32);
              v27 = v49;
            }
            else
            {
              si128 = _mm_load_si128((const __m128i *)&xmmword_42D0980);
              *(_BYTE *)(v26 + 24) = 32;
              *(_QWORD *)(v26 + 16) = 0x3A656D616E207265LL;
              *(__m128i *)v26 = si128;
              v29 = (__m128i *)(v24[4] + 25);
              v24[4] = (__int64)v29;
            }
            v30 = *(_QWORD *)(v27 + 24) - (_QWORD)v29;
            if ( v30 < (unsigned __int64)v20 )
            {
              a2 = (__int64)v19;
              v48 = sub_CB6200(v27, v19, (size_t)v20);
              v29 = *(__m128i **)(v48 + 32);
              v27 = v48;
              v30 = *(_QWORD *)(v48 + 24) - (_QWORD)v29;
            }
            else if ( v20 )
            {
              a2 = (__int64)v19;
              memcpy(v29, v19, (size_t)v20);
              v50 = *(_QWORD *)(v27 + 24);
              v29 = (__m128i *)&v20[*(_QWORD *)(v27 + 32)];
              *(_QWORD *)(v27 + 32) = v29;
              v30 = v50 - (_QWORD)v29;
            }
            if ( v30 <= 0x25 )
            {
              a2 = (__int64)"  Must match [a-zA-Z$_][a-zA-Z$_0-9]*\n";
              sub_CB6200(v27, "  Must match [a-zA-Z$_][a-zA-Z$_0-9]*\n", 0x26u);
            }
            else
            {
              v31 = _mm_load_si128((const __m128i *)&xmmword_42D0990);
              v29[2].m128i_i32[0] = 1564028208;
              v29[2].m128i_i16[2] = 2602;
              *v29 = v31;
              v29[1] = _mm_load_si128((const __m128i *)&xmmword_42D09A0);
              *(_QWORD *)(v27 + 32) += 38LL;
            }
LABEL_45:
            sub_2C76240(a1, a2, v26, v25);
            goto LABEL_4;
          }
LABEL_5:
          if ( v7 == 6 )
          {
            v13 = sub_BD5D20(v2);
            if ( v14 == 9 && *(_QWORD *)v13 == 0x6573752E6D766C6CLL && v13[8] == 100 )
              goto LABEL_64;
            v8 = (const char *)v2;
            v15 = sub_2C76620(a1, v2, 0);
            v16 = (__m128i *)v15[4];
            if ( (unsigned __int64)(v15[3] - (_QWORD)v16) <= 0x23 )
            {
              v8 = "appending linkage is not supported.\n";
              sub_CB6200((__int64)v15, "appending linkage is not supported.\n", 0x24u);
            }
            else
            {
              v17 = _mm_load_si128((const __m128i *)&xmmword_42D09C0);
              v16[2].m128i_i32[0] = 170812517;
              *v16 = v17;
              v16[1] = _mm_load_si128((const __m128i *)&xmmword_42D09D0);
              v15[4] += 36;
            }
          }
          else
          {
            if ( v7 != 9 )
            {
              if ( ((v6 >> 4) & 3u) - 1 > 1 )
              {
LABEL_8:
                sub_2C795F0(a1, v2);
                return;
              }
LABEL_46:
              v32 = sub_2C76620(a1, v2, 2u);
              v33 = (__m128i *)v32[4];
              if ( (unsigned __int64)(v32[3] - (_QWORD)v33) <= 0x2E )
              {
                sub_CB6200((__int64)v32, "Hidden/protected visibility flags are ignored.\n", 0x2Fu);
              }
              else
              {
                v34 = _mm_load_si128((const __m128i *)&xmmword_42D0A00);
                qmemcpy(&v33[2], "s are ignored.\n", 15);
                *v33 = v34;
                v33[1] = _mm_load_si128((const __m128i *)&xmmword_42D0A10);
                v32[4] += 47;
              }
              goto LABEL_8;
            }
            v8 = (const char *)v2;
            v9 = sub_2C76620(a1, v2, 0);
            v10 = (__m128i *)v9[4];
            if ( (unsigned __int64)(v9[3] - (_QWORD)v10) <= 0x25 )
            {
              v8 = "extern_weak linkage is not supported.\n";
              sub_CB6200((__int64)v9, "extern_weak linkage is not supported.\n", 0x26u);
            }
            else
            {
              v11 = _mm_load_si128((const __m128i *)&xmmword_42D09E0);
              v12 = 2606;
              v10[2].m128i_i32[0] = 1684370546;
              v10[2].m128i_i16[2] = 2606;
              *v10 = v11;
              v10[1] = _mm_load_si128((const __m128i *)&xmmword_42D09F0);
              v9[4] += 38;
            }
          }
          v35 = *(_BYTE **)(a1 + 16);
          if ( v35 )
            *v35 = 0;
          v36 = *(unsigned int *)(a1 + 4);
          if ( !(_DWORD)v36 )
          {
            v37 = *(_QWORD *)(a1 + 24);
            if ( *(_QWORD *)(v37 + 32) != *(_QWORD *)(v37 + 16) )
            {
              sub_CB5AE0((__int64 *)v37);
              v37 = *(_QWORD *)(a1 + 24);
            }
            sub_CEB520(*(_QWORD **)(v37 + 48), (__int64)v8, v36, (char *)v12);
          }
LABEL_64:
          if ( ((*(_BYTE *)(v2 + 32) >> 4) & 3u) - 1 > 1 )
            goto LABEL_8;
          goto LABEL_46;
        }
        v60 = (__int64)v62;
        if ( !v19 )
        {
          v61 = 0;
          LOBYTE(v62[0]) = 0;
          goto LABEL_80;
        }
        v57 = v20;
        if ( (unsigned __int64)v20 > 0xF )
        {
          v60 = sub_22409D0((__int64)&v60, (unsigned __int64 *)&v57, 0);
          v51 = (_QWORD *)v60;
          v62[0] = v57;
        }
        else
        {
          if ( v20 == (_BYTE *)1 )
          {
            v39 = v62;
            LOBYTE(v62[0]) = *v19;
            v40 = 1;
LABEL_79:
            v61 = v40;
            *((_BYTE *)v39 + v40) = 0;
LABEL_80:
            v41 = sub_22417D0(&v60, 0, 0);
            if ( (_QWORD *)v60 != v62 )
              j_j___libc_free_0(v60);
            if ( v41 == (char *)-1LL )
              goto LABEL_4;
            v42 = sub_2C76620(a1, a2, 0);
            v26 = v42[4];
            v43 = (__int64)v42;
            if ( (unsigned __int64)(v42[3] - v26) <= 0x18 )
            {
              v53 = sub_CB6200((__int64)v42, "Invalid identifier name: ", 0x19u);
              v45 = *(__m128i **)(v53 + 32);
              v43 = v53;
            }
            else
            {
              v44 = _mm_load_si128((const __m128i *)&xmmword_42D0980);
              *(_BYTE *)(v26 + 24) = 32;
              *(_QWORD *)(v26 + 16) = 0x3A656D616E207265LL;
              *(__m128i *)v26 = v44;
              v45 = (__m128i *)(v42[4] + 25);
              v42[4] = (__int64)v45;
            }
            v46 = *(_QWORD *)(v43 + 24) - (_QWORD)v45;
            if ( v46 < (unsigned __int64)v20 )
            {
              v52 = sub_CB6200(v43, v19, (size_t)v20);
              v45 = *(__m128i **)(v52 + 32);
              v43 = v52;
              v46 = *(_QWORD *)(v52 + 24) - (_QWORD)v45;
            }
            else if ( v20 )
            {
              memcpy(v45, v19, (size_t)v20);
              v54 = *(_QWORD *)(v43 + 24);
              v45 = (__m128i *)&v20[*(_QWORD *)(v43 + 32)];
              *(_QWORD *)(v43 + 32) = v45;
              v46 = v54 - (_QWORD)v45;
            }
            if ( v46 <= 0x1E )
            {
              a2 = (__int64)"  may not have null character.\n";
              sub_CB6200(v43, "  may not have null character.\n", 0x1Fu);
            }
            else
            {
              v47 = _mm_load_si128((const __m128i *)&xmmword_42D09B0);
              a2 = 11890;
              qmemcpy(&v45[1], "ull character.\n", 15);
              *v45 = v47;
              *(_QWORD *)(v43 + 32) += 31LL;
            }
            goto LABEL_45;
          }
          if ( !v20 )
          {
            v39 = v62;
            v40 = 0;
            goto LABEL_79;
          }
          v51 = v62;
        }
        memcpy(v51, v19, (size_t)v20);
        v40 = (__int64)v57;
        v39 = (_QWORD *)v60;
        goto LABEL_79;
      }
      goto LABEL_55;
    }
    v60 = v18;
    if ( v18 > 0xF )
    {
      v57 = (_QWORD *)sub_22409D0((__int64)&v57, (unsigned __int64 *)&v60, 0);
      dest[0] = v60;
      memcpy(v57, v19, (size_t)v20);
      v58 = v60;
      *((_BYTE *)v57 + v60) = 0;
      if ( sub_22416F0((__int64 *)&v57, (char *)"llvm.", 0, 5u) )
      {
        v56 = v20;
        v60 = (__int64)v62;
        v60 = sub_22409D0((__int64)&v60, (unsigned __int64 *)&v56, 0);
        v38 = (_QWORD *)v60;
        v62[0] = v56;
        goto LABEL_74;
      }
LABEL_55:
      if ( v57 == dest )
        goto LABEL_4;
      j_j___libc_free_0((unsigned __int64)v57);
      v6 = *(_BYTE *)(a2 + 32);
      v7 = v6 & 0xF;
      goto LABEL_5;
    }
    if ( v18 == 1 )
    {
      LOBYTE(dest[0]) = *v19;
    }
    else if ( v18 )
    {
      memcpy(dest, v19, v18);
      v58 = v60;
      *((_BYTE *)v57 + v60) = 0;
LABEL_19:
      if ( sub_22416F0((__int64 *)&v57, (char *)"llvm.", 0, 5u) )
      {
        v56 = v20;
        v60 = (__int64)v62;
        if ( v20 == (_BYTE *)1 )
        {
          LOBYTE(v62[0]) = *v19;
LABEL_22:
          v61 = (unsigned __int64)v56;
          v56[v60] = 0;
          v21 = sub_22416F0(&v60, "nvvm.", 0, 5u);
          goto LABEL_23;
        }
        if ( !v20 )
          goto LABEL_22;
        v38 = v62;
LABEL_74:
        memcpy(v38, v19, (size_t)v20);
        goto LABEL_22;
      }
      goto LABEL_55;
    }
    v58 = v18;
    *((_BYTE *)dest + v18) = 0;
    goto LABEL_19;
  }
}
