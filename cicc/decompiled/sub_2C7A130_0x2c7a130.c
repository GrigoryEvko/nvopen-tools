// Function: sub_2C7A130
// Address: 0x2c7a130
//
void __fastcall sub_2C7A130(__int64 a1, char *a2, __int64 a3)
{
  __int64 v4; // r12
  unsigned int v5; // eax
  __int64 v6; // rdi
  const char *v7; // rsi
  __int64 v8; // rax
  __m128i *v9; // rdx
  __m128i v10; // xmm0
  __int64 v11; // rcx
  _BYTE *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rax
  char *v16; // rcx
  __m128i *v17; // rdx
  __m128i v18; // xmm0
  __int64 v19; // rsi
  const char *v20; // rax
  __int64 v21; // rdx
  const char *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __m128i v25; // xmm0
  __int64 v26; // rdi
  __m128i *v27; // rax
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  char *v31; // rcx
  __m128i *v32; // rdx
  __int64 v33; // rdi
  __m128i si128; // xmm0
  void *v35; // rdx
  unsigned __int8 *v36; // r14
  const char *v37; // rsi
  __int64 v38; // rax
  char *v39; // rcx
  __int64 v40; // rdx
  __m128i v41; // xmm0
  const char *v42; // rsi
  __int64 v43; // rax
  char *v44; // rcx
  __int64 v45; // rdx
  __m128i v46; // xmm0
  _BYTE *v47; // rax
  __int64 v48; // rdi
  __int64 v49; // rax
  __m128i v50; // xmm0
  unsigned __int64 v51; // rdx
  __int64 v52; // rax
  _BYTE *v53; // rax
  __int64 v54; // rdi
  __int64 v55; // rax
  __m128i v56; // xmm0
  __m128i v57; // xmm0
  unsigned __int64 v58; // rdx
  _BYTE *v59; // r15
  size_t v60; // r14
  unsigned __int64 v61; // rax
  _QWORD *v62; // rdx
  _QWORD *v63; // r8
  const char *v64; // rsi
  __int64 v65; // rax
  char *v66; // rcx
  __int64 v67; // rdx
  __m128i v68; // xmm0
  _QWORD *v69; // rdi
  unsigned __int64 v70; // [rsp+8h] [rbp-58h] BYREF
  _QWORD *v71; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 v72; // [rsp+18h] [rbp-48h]
  _QWORD v73[8]; // [rsp+20h] [rbp-40h] BYREF

  v4 = (__int64)a2;
  if ( (a2[35] & 4) != 0 )
  {
    v28 = sub_B31D10((__int64)a2, (__int64)a2, a3);
    if ( v29 == 13 )
    {
      v29 = 0x74656D2E6D766C6CLL;
      if ( *(_QWORD *)v28 == 0x74656D2E6D766C6CLL && *(_DWORD *)(v28 + 8) == 1952539745 && *(_BYTE *)(v28 + 12) == 97 )
        return;
    }
    if ( ((*((_WORD *)a2 + 17) >> 1) & 0x200) != 0 )
    {
      if ( *(_DWORD *)(*((_QWORD *)a2 + 1) + 8LL) >> 8 != 4 )
      {
        v30 = sub_2C767C0(a1, (__int64)a2, 0);
        v32 = *(__m128i **)(v30 + 32);
        v33 = v30;
        if ( *(_QWORD *)(v30 + 24) - (_QWORD)v32 <= 0x30u )
        {
          a2 = "Explicit section marker other than llvm.metadata ";
          v52 = sub_CB6200(v30, "Explicit section marker other than llvm.metadata ", 0x31u);
          v35 = *(void **)(v52 + 32);
          v33 = v52;
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_42D0680);
          v32[3].m128i_i8[0] = 32;
          *v32 = si128;
          v32[1] = _mm_load_si128((const __m128i *)&xmmword_42D0A40);
          v32[2] = _mm_load_si128((const __m128i *)&xmmword_42D0A50);
          v35 = (void *)(*(_QWORD *)(v30 + 32) + 49LL);
          *(_QWORD *)(v30 + 32) = v35;
        }
        if ( *(_QWORD *)(v33 + 24) - (_QWORD)v35 <= 0xDu )
        {
          a2 = "is not allowed";
          sub_CB6200(v33, (unsigned __int8 *)"is not allowed", 0xEu);
        }
        else
        {
          qmemcpy(v35, "is not allowed", 14);
          *(_QWORD *)(v33 + 32) += 14LL;
        }
        v53 = *(_BYTE **)(a1 + 16);
        if ( v53 )
          *v53 = 0;
        if ( !*(_DWORD *)(a1 + 4) )
        {
          v54 = *(_QWORD *)(a1 + 24);
          if ( *(_QWORD *)(v54 + 32) != *(_QWORD *)(v54 + 16) )
          {
            sub_CB5AE0((__int64 *)v54);
            v54 = *(_QWORD *)(a1 + 24);
          }
          sub_CEB520(*(_QWORD **)(v54 + 48), (__int64)a2, (__int64)v35, v31);
        }
        goto LABEL_2;
      }
      v59 = (_BYTE *)sub_B31D10((__int64)a2, (__int64)a2, v29);
      v60 = v58;
      if ( !v59 )
      {
        LOBYTE(v73[0]) = 0;
        v71 = v73;
        v72 = 0;
        if ( !memcmp(v73, ".nv.constant", 0xCu) )
          goto LABEL_2;
LABEL_88:
        v64 = (const char *)v4;
        v65 = sub_2C767C0(a1, v4, 0);
        v67 = *(_QWORD *)(v65 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(v65 + 24) - v67) <= 0x30 )
        {
          v64 = "Explicit section on constant is not constant bank";
          sub_CB6200(v65, "Explicit section on constant is not constant bank", 0x31u);
        }
        else
        {
          v68 = _mm_load_si128((const __m128i *)&xmmword_42D0680);
          *(_BYTE *)(v67 + 48) = 107;
          *(__m128i *)v67 = v68;
          *(__m128i *)(v67 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D0A20);
          *(__m128i *)(v67 + 32) = _mm_load_si128((const __m128i *)&xmmword_42D0A30);
          *(_QWORD *)(v65 + 32) += 49LL;
        }
        sub_2C76240(a1, (__int64)v64, v67, v66);
        v63 = v71;
LABEL_84:
        if ( v63 != v73 )
          j_j___libc_free_0((unsigned __int64)v63);
        goto LABEL_2;
      }
      v70 = v58;
      v61 = v58;
      v71 = v73;
      if ( v58 > 0xF )
      {
        v71 = (_QWORD *)sub_22409D0((__int64)&v71, &v70, 0);
        v69 = v71;
        v73[0] = v70;
      }
      else
      {
        if ( v58 == 1 )
        {
          LOBYTE(v73[0]) = *v59;
          v62 = v73;
          goto LABEL_83;
        }
        if ( !v58 )
        {
          v62 = v73;
LABEL_83:
          v72 = v61;
          *((_BYTE *)v62 + v61) = 0;
          v63 = v71;
          if ( !memcmp(v71, ".nv.constant", 0xCu) )
            goto LABEL_84;
          goto LABEL_88;
        }
        v69 = v73;
      }
      memcpy(v69, v59, v60);
      v61 = v70;
      v62 = v71;
      goto LABEL_83;
    }
  }
LABEL_2:
  if ( (*(_BYTE *)(v4 + 7) & 0x10) == 0 )
    goto LABEL_3;
  v20 = sub_BD5D20(v4);
  if ( v21 == 17
    && !(*(_QWORD *)v20 ^ 0x6F6C672E6D766C6CLL | *((_QWORD *)v20 + 1) ^ 0x726F74635F6C6162LL)
    && v20[16] == 115 )
  {
    v49 = sub_2C767C0(a1, v4, 1u);
    v25 = _mm_load_si128((const __m128i *)&xmmword_3F89B40);
    v26 = v49;
    v27 = *(__m128i **)(v49 + 32);
    if ( *(_QWORD *)(v26 + 24) - (_QWORD)v27 > 0x23u )
    {
LABEL_60:
      *v27 = v25;
      v50 = _mm_load_si128((const __m128i *)&xmmword_42D0A60);
      v27[2].m128i_i32[0] = 170812517;
      v27[1] = v50;
      *(_QWORD *)(v26 + 32) += 36LL;
      goto LABEL_3;
    }
    sub_CB6200(v26, "llvm.global_ctors is not supported.\n", 0x24u);
  }
  else
  {
    v22 = sub_BD5D20(v4);
    if ( v23 == 17
      && !(*(_QWORD *)v22 ^ 0x6F6C672E6D766C6CLL | *((_QWORD *)v22 + 1) ^ 0x726F74645F6C6162LL)
      && v22[16] == 115 )
    {
      v24 = sub_2C767C0(a1, v4, 1u);
      v25 = _mm_load_si128((const __m128i *)&xmmword_3F89B30);
      v26 = v24;
      v27 = *(__m128i **)(v24 + 32);
      if ( *(_QWORD *)(v26 + 24) - (_QWORD)v27 <= 0x23u )
      {
        sub_CB6200(v26, "llvm.global_dtors is not supported.\n", 0x24u);
        goto LABEL_3;
      }
      goto LABEL_60;
    }
  }
LABEL_3:
  v5 = *(_DWORD *)(*(_QWORD *)(v4 + 8) + 8LL) >> 8;
  if ( *(_DWORD *)(*(_QWORD *)(v4 + 8) + 8LL) > 0x4FFu )
  {
    if ( v5 != 5 )
    {
LABEL_48:
      v42 = (const char *)v4;
      v43 = sub_2C767C0(a1, v4, 0);
      v45 = *(_QWORD *)(v43 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v43 + 24) - v45) <= 0x29 )
      {
        v42 = "Invalid address space for global variable\n";
        sub_CB6200(v43, "Invalid address space for global variable\n", 0x2Au);
      }
      else
      {
        v46 = _mm_load_si128((const __m128i *)&xmmword_42D0500);
        qmemcpy((void *)(v45 + 32), " variable\n", 10);
        *(__m128i *)v45 = v46;
        *(__m128i *)(v45 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D0510);
        *(_QWORD *)(v43 + 32) += 42LL;
      }
      v47 = *(_BYTE **)(a1 + 16);
      if ( v47 )
        *v47 = 0;
      if ( !*(_DWORD *)(a1 + 4) )
      {
        v48 = *(_QWORD *)(a1 + 24);
        if ( *(_QWORD *)(v48 + 32) != *(_QWORD *)(v48 + 16) )
        {
          sub_CB5AE0((__int64 *)v48);
          v48 = *(_QWORD *)(a1 + 24);
        }
        sub_CEB520(*(_QWORD **)(v48 + 48), (__int64)v42, v45, v44);
      }
LABEL_7:
      if ( (unsigned __int8)sub_CE8750((_BYTE *)v4) )
        goto LABEL_8;
      goto LABEL_25;
    }
    if ( *(_DWORD *)a1 == 1 )
      goto LABEL_7;
    goto LABEL_22;
  }
  if ( v5 <= 2 )
  {
    if ( v5 )
    {
      if ( v5 == 1 )
        goto LABEL_7;
      goto LABEL_48;
    }
    if ( *(_DWORD *)a1 != 1 )
      goto LABEL_7;
LABEL_22:
    v15 = sub_2C767C0(a1, v4, 0);
    v17 = *(__m128i **)(v15 + 32);
    if ( *(_QWORD *)(v15 + 24) - (_QWORD)v17 <= 0x29u )
    {
      v19 = (__int64)"Invalid address space for global variable\n";
      sub_CB6200(v15, "Invalid address space for global variable\n", 0x2Au);
    }
    else
    {
      v18 = _mm_load_si128((const __m128i *)&xmmword_42D0500);
      v19 = 2661;
      qmemcpy(&v17[2], " variable\n", 10);
      *v17 = v18;
      v17[1] = _mm_load_si128((const __m128i *)&xmmword_42D0510);
      *(_QWORD *)(v15 + 32) += 42LL;
    }
    goto LABEL_24;
  }
  if ( !(unsigned __int8)sub_CE8750((_BYTE *)v4) && !(unsigned __int8)sub_CE87C0((_BYTE *)v4) )
    goto LABEL_7;
  v19 = v4;
  v55 = sub_2C767C0(a1, v4, 0);
  v17 = *(__m128i **)(v55 + 32);
  if ( *(_QWORD *)(v55 + 24) - (_QWORD)v17 <= 0x36u )
  {
    v19 = (__int64)"Texture/surface variables must be global address space\n";
    sub_CB6200(v55, "Texture/surface variables must be global address space\n", 0x37u);
  }
  else
  {
    v56 = _mm_load_si128((const __m128i *)&xmmword_42D0A70);
    v17[3].m128i_i32[0] = 1634759456;
    v17[3].m128i_i16[2] = 25955;
    *v17 = v56;
    v57 = _mm_load_si128((const __m128i *)&xmmword_42D0A80);
    v17[3].m128i_i8[6] = 10;
    v17[1] = v57;
    v17[2] = _mm_load_si128((const __m128i *)&xmmword_42D0A90);
    *(_QWORD *)(v55 + 32) += 55LL;
  }
LABEL_24:
  sub_2C76240(a1, v19, (__int64)v17, v16);
  if ( (unsigned __int8)sub_CE8750((_BYTE *)v4) )
  {
LABEL_8:
    if ( *(_BYTE *)(*(_QWORD *)(v4 + 8) + 8LL) != 14 )
      goto LABEL_10;
    goto LABEL_9;
  }
LABEL_25:
  if ( (unsigned __int8)sub_CE87C0((_BYTE *)v4) )
  {
    if ( *(_BYTE *)(*(_QWORD *)(v4 + 8) + 8LL) != 14 )
      goto LABEL_10;
LABEL_9:
    v6 = *(_QWORD *)(v4 + 24);
    if ( *(_BYTE *)(v6 + 8) == 12 )
    {
      v71 = (_QWORD *)sub_BCAE30(v6);
      v72 = v51;
      if ( sub_CA1930(&v71) == 64 )
        goto LABEL_18;
    }
LABEL_10:
    v7 = (const char *)v4;
    v8 = sub_2C767C0(a1, v4, 0);
    v9 = *(__m128i **)(v8 + 32);
    if ( *(_QWORD *)(v8 + 24) - (_QWORD)v9 <= 0x2Eu )
    {
      v7 = "Texture and surface variables must be type i64*";
      sub_CB6200(v8, "Texture and surface variables must be type i64*", 0x2Fu);
    }
    else
    {
      v10 = _mm_load_si128((const __m128i *)&xmmword_42D0AA0);
      v11 = 13366;
      qmemcpy(&v9[2], "st be type i64*", 15);
      *v9 = v10;
      v9[1] = _mm_load_si128((const __m128i *)&xmmword_42D0AB0);
      *(_QWORD *)(v8 + 32) += 47LL;
    }
    v12 = *(_BYTE **)(a1 + 16);
    if ( v12 )
      *v12 = 0;
    v13 = *(unsigned int *)(a1 + 4);
    if ( !(_DWORD)v13 )
    {
      v14 = *(_QWORD *)(a1 + 24);
      if ( *(_QWORD *)(v14 + 32) != *(_QWORD *)(v14 + 16) )
      {
        sub_CB5AE0((__int64 *)v14);
        v14 = *(_QWORD *)(a1 + 24);
      }
      sub_CEB520(*(_QWORD **)(v14 + 48), (__int64)v7, v13, (char *)v11);
    }
  }
LABEL_18:
  if ( !sub_B2FC80(v4) )
  {
    v36 = *(unsigned __int8 **)(v4 - 32);
    sub_2C77070(a1, (__int64)v36, v4);
    if ( *(_DWORD *)(*(_QWORD *)(v4 + 8) + 8LL) >> 8 == 3 && (unsigned int)*v36 - 12 > 1 )
    {
      v37 = (const char *)v4;
      v38 = sub_2C767C0(a1, v4, 0);
      v40 = *(_QWORD *)(v38 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v38 + 24) - v40) <= 0x25 )
      {
        v37 = "Shared variables can't be initialized\n";
        sub_CB6200(v38, "Shared variables can't be initialized\n", 0x26u);
      }
      else
      {
        v41 = _mm_load_si128((const __m128i *)&xmmword_42D0AC0);
        *(_DWORD *)(v40 + 32) = 1702521196;
        *(_WORD *)(v40 + 36) = 2660;
        *(__m128i *)v40 = v41;
        *(__m128i *)(v40 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D0AD0);
        *(_QWORD *)(v38 + 32) += 38LL;
      }
      if ( *(_DWORD *)a1 != 1 )
        sub_2C76240(a1, (__int64)v37, v40, v39);
    }
  }
  sub_2C797D0(a1, v4);
}
