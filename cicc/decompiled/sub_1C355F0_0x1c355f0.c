// Function: sub_1C355F0
// Address: 0x1c355f0
//
const char *__fastcall sub_1C355F0(_DWORD *a1, _BYTE *a2)
{
  unsigned int v4; // eax
  __int64 v5; // rdi
  __int64 v6; // rax
  __m128i *v7; // rdx
  __m128i v8; // xmm0
  __int64 v9; // rax
  __m128i *v10; // rdx
  __m128i v11; // xmm0
  __int64 v12; // r14
  __int64 v13; // rax
  __m128i *v14; // rdx
  __m128i v15; // xmm0
  const char *result; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __m128i *v19; // rdx
  __int64 v20; // rdi
  __m128i si128; // xmm0
  void *v22; // rdx
  const char *v23; // rax
  __int64 v24; // rdx
  const char *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __m128i *v28; // rdx
  __m128i v29; // xmm0
  __int64 v30; // rdi
  __m128i *v31; // rax
  __m128i v32; // xmm0
  __int64 v33; // rax
  __int64 v34; // rax
  __m128i *v35; // rdx
  __m128i v36; // xmm0
  __m128i v37; // xmm0
  unsigned __int64 v38; // rdx
  _BYTE *v39; // r15
  size_t v40; // r14
  unsigned __int64 v41; // rax
  _QWORD *v42; // rdx
  _QWORD *v43; // r8
  __int64 v44; // rax
  __m128i *v45; // rdx
  __m128i v46; // xmm0
  _QWORD *v47; // rdi
  unsigned __int64 v48; // [rsp+8h] [rbp-58h] BYREF
  _QWORD *v49; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 v50; // [rsp+18h] [rbp-48h]
  _QWORD v51[8]; // [rsp+20h] [rbp-40h] BYREF

  if ( (a2[34] & 0x20) == 0 )
    goto LABEL_2;
  result = (const char *)sub_15E61A0((__int64)a2);
  if ( v17 != 13
    || *(_QWORD *)result != 0x74656D2E6D766C6CLL
    || *((_DWORD *)result + 2) != 1952539745
    || result[12] != 97 )
  {
    if ( (a2[34] & 0x20) == 0 )
      goto LABEL_2;
    if ( *(_DWORD *)(*(_QWORD *)a2 + 8LL) >> 8 != 4 )
    {
      v18 = sub_1C31E60((__int64)a1, (__int64)a2, 0);
      v19 = *(__m128i **)(v18 + 24);
      v20 = v18;
      if ( *(_QWORD *)(v18 + 16) - (_QWORD)v19 <= 0x30u )
      {
        v33 = sub_16E7EE0(v18, "Explicit section marker other than llvm.metadata ", 0x31u);
        v22 = *(void **)(v33 + 24);
        v20 = v33;
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_42D0680);
        v19[3].m128i_i8[0] = 32;
        *v19 = si128;
        v19[1] = _mm_load_si128((const __m128i *)&xmmword_42D0A40);
        v19[2] = _mm_load_si128((const __m128i *)&xmmword_42D0A50);
        v22 = (void *)(*(_QWORD *)(v18 + 24) + 49LL);
        *(_QWORD *)(v18 + 24) = v22;
      }
      if ( *(_QWORD *)(v20 + 16) - (_QWORD)v22 <= 0xDu )
      {
        sub_16E7EE0(v20, "is not allowed", 0xEu);
      }
      else
      {
        qmemcpy(v22, "is not allowed", 14);
        *(_QWORD *)(v20 + 24) += 14LL;
      }
      sub_1C31880((__int64)a1);
      if ( (a2[23] & 0x20) == 0 )
        goto LABEL_3;
      goto LABEL_39;
    }
    v39 = (_BYTE *)sub_15E61A0((__int64)a2);
    v40 = v38;
    if ( !v39 )
    {
      LOBYTE(v51[0]) = 0;
      v49 = v51;
      v50 = 0;
      if ( !memcmp(v51, ".nv.constant", 0xCu) )
        goto LABEL_2;
LABEL_69:
      v44 = sub_1C31E60((__int64)a1, (__int64)a2, 0);
      v45 = *(__m128i **)(v44 + 24);
      if ( *(_QWORD *)(v44 + 16) - (_QWORD)v45 <= 0x30u )
      {
        sub_16E7EE0(v44, "Explicit section on constant is not constant bank", 0x31u);
      }
      else
      {
        v46 = _mm_load_si128((const __m128i *)&xmmword_42D0680);
        v45[3].m128i_i8[0] = 107;
        *v45 = v46;
        v45[1] = _mm_load_si128((const __m128i *)&xmmword_42D0A20);
        v45[2] = _mm_load_si128((const __m128i *)&xmmword_42D0A30);
        *(_QWORD *)(v44 + 24) += 49LL;
      }
      sub_1C31880((__int64)a1);
      v43 = v49;
LABEL_62:
      if ( v43 != v51 )
        j_j___libc_free_0(v43, v51[0] + 1LL);
LABEL_2:
      if ( (a2[23] & 0x20) == 0 )
        goto LABEL_3;
LABEL_39:
      v23 = sub_1649960((__int64)a2);
      if ( v24 == 17
        && !(*(_QWORD *)v23 ^ 0x6F6C672E6D766C6CLL | *((_QWORD *)v23 + 1) ^ 0x726F74635F6C6162LL)
        && v23[16] == 115 )
      {
        v30 = sub_1C31E60((__int64)a1, (__int64)a2, 1u);
        v31 = *(__m128i **)(v30 + 24);
        if ( *(_QWORD *)(v30 + 16) - (_QWORD)v31 <= 0x23u )
        {
          sub_16E7EE0(v30, "llvm.global_ctors is not supported.\n", 0x24u);
        }
        else
        {
          v32 = _mm_load_si128((const __m128i *)&xmmword_3F89B40);
          v31[2].m128i_i32[0] = 170812517;
          *v31 = v32;
          v31[1] = _mm_load_si128((const __m128i *)&xmmword_42D0A60);
          *(_QWORD *)(v30 + 24) += 36LL;
        }
      }
      else
      {
        v25 = sub_1649960((__int64)a2);
        if ( v26 == 17
          && !(*(_QWORD *)v25 ^ 0x6F6C672E6D766C6CLL | *((_QWORD *)v25 + 1) ^ 0x726F74645F6C6162LL)
          && v25[16] == 115 )
        {
          v27 = sub_1C31E60((__int64)a1, (__int64)a2, 1u);
          v28 = *(__m128i **)(v27 + 24);
          if ( *(_QWORD *)(v27 + 16) - (_QWORD)v28 <= 0x23u )
          {
            sub_16E7EE0(v27, "llvm.global_dtors is not supported.\n", 0x24u);
          }
          else
          {
            v29 = _mm_load_si128((const __m128i *)&xmmword_3F89B30);
            v28[2].m128i_i32[0] = 170812517;
            *v28 = v29;
            v28[1] = _mm_load_si128((const __m128i *)&xmmword_42D0A60);
            *(_QWORD *)(v27 + 24) += 36LL;
          }
        }
      }
LABEL_3:
      v4 = *(_DWORD *)(*(_QWORD *)a2 + 8LL) >> 8;
      if ( *(_DWORD *)(*(_QWORD *)a2 + 8LL) > 0x4FFu )
      {
        if ( v4 != 5 || *a1 != 1 )
          goto LABEL_15;
      }
      else if ( v4 > 2 )
      {
        if ( (unsigned __int8)sub_1C2E830((__int64)a2) || (unsigned __int8)sub_1C2E860((__int64)a2) )
        {
          v34 = sub_1C31E60((__int64)a1, (__int64)a2, 0);
          v35 = *(__m128i **)(v34 + 24);
          if ( *(_QWORD *)(v34 + 16) - (_QWORD)v35 <= 0x36u )
          {
            sub_16E7EE0(v34, "Texture/surface variables must be global address space\n", 0x37u);
          }
          else
          {
            v36 = _mm_load_si128((const __m128i *)&xmmword_42D0A70);
            v35[3].m128i_i32[0] = 1634759456;
            v35[3].m128i_i16[2] = 25955;
            *v35 = v36;
            v37 = _mm_load_si128((const __m128i *)&xmmword_42D0A80);
            v35[3].m128i_i8[6] = 10;
            v35[1] = v37;
            v35[2] = _mm_load_si128((const __m128i *)&xmmword_42D0A90);
            *(_QWORD *)(v34 + 24) += 55LL;
          }
LABEL_17:
          sub_1C31880((__int64)a1);
          if ( (unsigned __int8)sub_1C2E830((__int64)a2) )
            goto LABEL_8;
LABEL_18:
          if ( !(unsigned __int8)sub_1C2E860((__int64)a2) )
          {
            if ( sub_15E4F60((__int64)a2) )
              return sub_1C34CC0(a1, (__int64)a2);
LABEL_20:
            v12 = *((_QWORD *)a2 - 3);
            sub_1C320A0((__int64)a1, v12, (__int64)a2);
            if ( *(_DWORD *)(*(_QWORD *)a2 + 8LL) >> 8 == 3 && *(_BYTE *)(v12 + 16) != 9 )
            {
              v13 = sub_1C31E60((__int64)a1, (__int64)a2, 0);
              v14 = *(__m128i **)(v13 + 24);
              if ( *(_QWORD *)(v13 + 16) - (_QWORD)v14 <= 0x25u )
              {
                sub_16E7EE0(v13, "Shared variables can't be initialized\n", 0x26u);
              }
              else
              {
                v15 = _mm_load_si128((const __m128i *)&xmmword_42D0AC0);
                v14[2].m128i_i32[0] = 1702521196;
                v14[2].m128i_i16[2] = 2660;
                *v14 = v15;
                v14[1] = _mm_load_si128((const __m128i *)&xmmword_42D0AD0);
                *(_QWORD *)(v13 + 24) += 38LL;
              }
              if ( *a1 != 1 )
                sub_1C31880((__int64)a1);
            }
            return sub_1C34CC0(a1, (__int64)a2);
          }
LABEL_8:
          if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 15
            || (v5 = *((_QWORD *)a2 + 3), *(_BYTE *)(v5 + 8) != 11)
            || (unsigned int)sub_1643030(v5) != 64 )
          {
            v6 = sub_1C31E60((__int64)a1, (__int64)a2, 0);
            v7 = *(__m128i **)(v6 + 24);
            if ( *(_QWORD *)(v6 + 16) - (_QWORD)v7 <= 0x2Eu )
            {
              sub_16E7EE0(v6, "Texture and surface variables must be type i64*", 0x2Fu);
            }
            else
            {
              v8 = _mm_load_si128((const __m128i *)&xmmword_42D0AA0);
              qmemcpy(&v7[2], "st be type i64*", 15);
              *v7 = v8;
              v7[1] = _mm_load_si128((const __m128i *)&xmmword_42D0AB0);
              *(_QWORD *)(v6 + 24) += 47LL;
            }
            sub_1C31880((__int64)a1);
          }
          if ( sub_15E4F60((__int64)a2) )
            return sub_1C34CC0(a1, (__int64)a2);
          goto LABEL_20;
        }
      }
      else
      {
        if ( v4 )
        {
          if ( v4 == 1 )
            goto LABEL_7;
          goto LABEL_15;
        }
        if ( *a1 == 1 )
        {
LABEL_15:
          v9 = sub_1C31E60((__int64)a1, (__int64)a2, 0);
          v10 = *(__m128i **)(v9 + 24);
          if ( *(_QWORD *)(v9 + 16) - (_QWORD)v10 <= 0x29u )
          {
            sub_16E7EE0(v9, "Invalid address space for global variable\n", 0x2Au);
          }
          else
          {
            v11 = _mm_load_si128((const __m128i *)&xmmword_42D0500);
            qmemcpy(&v10[2], " variable\n", 10);
            *v10 = v11;
            v10[1] = _mm_load_si128((const __m128i *)&xmmword_42D0510);
            *(_QWORD *)(v9 + 24) += 42LL;
          }
          goto LABEL_17;
        }
      }
LABEL_7:
      if ( (unsigned __int8)sub_1C2E830((__int64)a2) )
        goto LABEL_8;
      goto LABEL_18;
    }
    v48 = v38;
    v41 = v38;
    v49 = v51;
    if ( v38 > 0xF )
    {
      v49 = (_QWORD *)sub_22409D0(&v49, &v48, 0);
      v47 = v49;
      v51[0] = v48;
    }
    else
    {
      if ( v38 == 1 )
      {
        LOBYTE(v51[0]) = *v39;
        v42 = v51;
        goto LABEL_61;
      }
      if ( !v38 )
      {
        v42 = v51;
        goto LABEL_61;
      }
      v47 = v51;
    }
    memcpy(v47, v39, v40);
    v41 = v48;
    v42 = v49;
LABEL_61:
    v50 = v41;
    *((_BYTE *)v42 + v41) = 0;
    v43 = v49;
    if ( !memcmp(v49, ".nv.constant", 0xCu) )
      goto LABEL_62;
    goto LABEL_69;
  }
  return result;
}
