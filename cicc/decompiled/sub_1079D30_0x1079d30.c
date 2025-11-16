// Function: sub_1079D30
// Address: 0x1079d30
//
__int64 __fastcall sub_1079D30(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        _QWORD *a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r15
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 *v19; // rax
  __int64 v20; // rdx
  _QWORD *v21; // rax
  char *v22; // rcx
  __int64 result; // rax
  char v24; // r8
  __int64 v25; // r9
  unsigned int v26; // eax
  _BYTE *v27; // r9
  unsigned __int64 v28; // r13
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  _QWORD *v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 *v37; // rdx
  void *v38; // rax
  __int64 v39; // rax
  _BYTE *v40; // r9
  __int64 v41; // rax
  int v42; // eax
  unsigned int v43; // esi
  __int64 v44; // r8
  int v45; // r10d
  _QWORD *v46; // rdi
  unsigned int v47; // edx
  __int64 v48; // rcx
  __m128i *v49; // rsi
  const __m128i **v50; // rdi
  __int64 v51; // rax
  __m128i *v52; // rsi
  __m128i *v53; // rsi
  int v54; // eax
  int v55; // edx
  __int64 v56; // rdx
  __int64 v57; // rdi
  unsigned int v58; // esi
  __int64 *v59; // rcx
  __int64 v60; // r9
  int v61; // ecx
  int v62; // r10d
  __int64 v63; // [rsp+8h] [rbp-148h]
  _BYTE *v64; // [rsp+8h] [rbp-148h]
  _DWORD *v65; // [rsp+10h] [rbp-140h]
  __int64 v67; // [rsp+18h] [rbp-138h]
  __int64 v68; // [rsp+18h] [rbp-138h]
  __int64 v69; // [rsp+20h] [rbp-130h]
  __int64 v70; // [rsp+28h] [rbp-128h]
  __m128i v71[2]; // [rsp+30h] [rbp-120h] BYREF
  char v72; // [rsp+50h] [rbp-100h]
  char v73; // [rsp+51h] [rbp-FFh]
  __m128i v74; // [rsp+60h] [rbp-F0h] BYREF
  __int16 v75; // [rsp+80h] [rbp-D0h]
  __m128i v76; // [rsp+90h] [rbp-C0h] BYREF
  _QWORD *v77; // [rsp+A0h] [rbp-B0h]
  __int64 v78; // [rsp+A8h] [rbp-A8h]
  __int16 v79; // [rsp+B0h] [rbp-A0h]
  __m128i v80[2]; // [rsp+C0h] [rbp-90h] BYREF
  char v81; // [rsp+E0h] [rbp-70h]
  char v82; // [rsp+E1h] [rbp-6Fh]
  __m128i v83; // [rsp+F0h] [rbp-60h] BYREF
  __m128i v84; // [rsp+100h] [rbp-50h] BYREF
  __int64 v85; // [rsp+110h] [rbp-40h]

  v12 = *(_QWORD *)(a3 + 8);
  v70 = a9;
  v13 = sub_E5C2C0((__int64)a2, a3);
  v16 = *a2;
  v69 = *(unsigned int *)(a4 + 8) + v13;
  if ( a8 )
  {
    v17 = *(_QWORD *)(a8 + 16);
    if ( (*(_BYTE *)(v12 + 48) & 0x10) != 0 )
    {
      v82 = 1;
      v80[0].m128i_i64[0] = (__int64)"' unsupported subtraction expression used in relocation in code section.";
      v81 = 3;
      if ( (*(_BYTE *)(v17 + 8) & 1) != 0 )
        goto LABEL_6;
      goto LABEL_10;
    }
    v18 = *(_QWORD *)v17;
    if ( *(_QWORD *)v17 )
    {
      if ( v12 != *(_QWORD *)(v18 + 8) )
        goto LABEL_5;
    }
    else
    {
      if ( (*(_BYTE *)(v17 + 9) & 0x70) != 0x20
        || *(char *)(v17 + 8) < 0
        || (*(_BYTE *)(v17 + 8) |= 8u, v34 = sub_E807D0(*(_QWORD *)(v17 + 24)), v18 = 0, (*(_QWORD *)v17 = v34) == 0) )
      {
        v82 = 1;
        v80[0].m128i_i64[0] = (__int64)"' can not be undefined in a subtraction expression";
        v36 = 0;
        v81 = 3;
        if ( (*(_BYTE *)(v17 + 8) & 1) != 0 )
        {
          v37 = *(__int64 **)(v17 - 8);
          v36 = *v37;
          v18 = (__int64)(v37 + 3);
        }
        v74.m128i_i64[0] = v18;
        v74.m128i_i64[1] = v36;
        v75 = 261;
        v73 = 1;
        v71[0].m128i_i64[0] = (__int64)"symbol '";
        v72 = 3;
        sub_9C6370(&v76, v71, &v74, 261, v14, v15);
        goto LABEL_8;
      }
      if ( v12 != v34[1] )
      {
LABEL_5:
        v82 = 1;
        v80[0].m128i_i64[0] = (__int64)"' can not be placed in a different section";
        v81 = 3;
        if ( (*(_BYTE *)(v17 + 8) & 1) != 0 )
        {
LABEL_6:
          v19 = *(__int64 **)(v17 - 8);
          v20 = *v19;
          v21 = v19 + 3;
LABEL_7:
          v77 = v21;
          v22 = "symbol '";
          v76.m128i_i64[0] = (__int64)"symbol '";
          v79 = 1283;
          v78 = v20;
LABEL_8:
          sub_9C6370(&v83, &v76, v80, (__int64)v22, v14, v15);
          return sub_E66880(v16, *(_QWORD **)(a4 + 16), (__int64)&v83);
        }
LABEL_10:
        v20 = 0;
        v21 = 0;
        goto LABEL_7;
      }
    }
    v35 = sub_E5C4C0((__int64)a2, v17);
    v24 = 1;
    v70 = v69 + v70 - v35;
  }
  else
  {
    v24 = 0;
  }
  v65 = (_DWORD *)a7;
  v25 = *(_QWORD *)(a7 + 16);
  result = *(_QWORD *)(v12 + 128);
  if ( *(_QWORD *)(v12 + 136) > 0xAu
    && *(_QWORD *)result == 0x72615F74696E692ELL
    && *(_WORD *)(result + 8) == 24946
    && *(_BYTE *)(result + 10) == 121 )
  {
    *(_BYTE *)(v25 + 44) = 1;
    return result;
  }
  if ( (*(_BYTE *)(v25 + 9) & 0x70) == 0x20 )
  {
    v51 = *(_QWORD *)(v25 + 24);
    *(_BYTE *)(v25 + 8) |= 8u;
    if ( *(_BYTE *)v51 == 2 && *(_WORD *)(v51 + 1) == 29 )
LABEL_103:
      BUG();
  }
  v63 = v25;
  *a5 = 0;
  v26 = (*(__int64 (__fastcall **)(_QWORD, __int64 *, __int64, __int64, _QWORD))(**(_QWORD **)(a1 + 112) + 24LL))(
          *(_QWORD *)(a1 + 112),
          &a7,
          a4,
          v12,
          v24 & 1);
  v27 = (_BYTE *)v63;
  v28 = v26;
  if ( v26 - 8 <= 1 || v26 == 22 )
  {
    v68 = v63;
    v38 = sub_1079CD0(v63, 1);
    v27 = (_BYTE *)v63;
    if ( v38 )
    {
      if ( !*(_BYTE *)(v12 + 174) )
        sub_C64ED0("relocations for function or section offsets are only supported in metadata sections", 1u);
      v39 = *((_QWORD *)sub_1079CD0(v63, 1) + 1);
      if ( (*(_BYTE *)(v39 + 48) & 0x10) == 0 )
      {
        v40 = *(_BYTE **)(v39 + 16);
        goto LABEL_41;
      }
      v56 = *(unsigned int *)(a1 + 424);
      v57 = *(_QWORD *)(a1 + 408);
      if ( (_DWORD)v56 )
      {
        v58 = (v56 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
        v59 = (__int64 *)(v57 + 16LL * v58);
        v60 = *v59;
        if ( v39 == *v59 )
        {
LABEL_89:
          if ( v59 != (__int64 *)(v57 + 16 * v56) )
          {
            v40 = (_BYTE *)v59[1];
LABEL_41:
            if ( !v40 )
              sub_C64ED0("section symbol is required for relocation", 1u);
            v64 = v40;
            v41 = sub_E5C4C0((__int64)a2, v68);
            v27 = v64;
            v70 += v41;
            if ( (unsigned int)v28 > 0x18 )
              goto LABEL_32;
            goto LABEL_18;
          }
        }
        else
        {
          v61 = 1;
          while ( v60 != -4096 )
          {
            v62 = v61 + 1;
            v58 = (v56 - 1) & (v61 + v58);
            v59 = (__int64 *)(v57 + 16LL * v58);
            v60 = *v59;
            if ( v39 == *v59 )
              goto LABEL_89;
            v61 = v62;
          }
        }
      }
      sub_C64ED0("section doesn't have defining symbol", 1u);
    }
  }
  if ( (unsigned int)v28 > 0x18 )
  {
LABEL_32:
    if ( (v27[8] & 1) == 0 || !**((_QWORD **)v27 - 1) )
      sub_C64ED0("relocations against un-named temporaries are not yet supported by wasm", 1u);
    v27[9] |= 8u;
    goto LABEL_49;
  }
LABEL_18:
  v29 = 17567750;
  if ( _bittest64(&v29, v28) )
  {
    v67 = (__int64)v27;
    v83.m128i_i64[0] = (__int64)"__indirect_function_table";
    LOWORD(v85) = 259;
    v30 = sub_E65280(v16, (const char **)&v83);
    if ( !v30 )
      sub_C64ED0("missing indirect function table symbol", 1u);
    if ( !*(_BYTE *)(v30 + 36) || *(_DWORD *)(v30 + 32) != 5 || !*(_BYTE *)(v30 + 176) || *(_DWORD *)(v30 + 136) != 112 )
      sub_C64ED0("__indirect_function_table symbol has wrong type", 1u);
    *(_WORD *)(v30 + 12) |= 0x80u;
    sub_E5CB20((__int64)a2, v30, v31, v32, v33, v67);
    v27 = (_BYTE *)v67;
  }
  if ( (_DWORD)v28 != 6 )
    goto LABEL_32;
LABEL_49:
  v42 = *v65 >> 8;
  if ( (_WORD)v42 == 1 || (_WORD)v42 == 128 )
    v27[45] = 1;
  v83.m128i_i64[1] = (__int64)v27;
  v84.m128i_i32[2] = v28;
  v83.m128i_i64[0] = v69;
  result = v70;
  v85 = v12;
  v84.m128i_i64[0] = v70;
  if ( *(_BYTE *)(v12 + 173) )
  {
    v52 = *(__m128i **)(a1 + 152);
    if ( v52 == *(__m128i **)(a1 + 160) )
    {
      return sub_1076F80((const __m128i **)(a1 + 144), v52, &v83);
    }
    else
    {
      if ( v52 )
      {
        *v52 = _mm_loadu_si128(&v83);
        v52[1] = _mm_loadu_si128(&v84);
        result = v85;
        v52[2].m128i_i64[0] = v85;
        v52 = *(__m128i **)(a1 + 152);
      }
      *(_QWORD *)(a1 + 152) = (char *)v52 + 40;
    }
  }
  else
  {
    if ( (*(_BYTE *)(v12 + 48) & 0x10) == 0 )
    {
      if ( !*(_BYTE *)(v12 + 174) )
        goto LABEL_103;
      v43 = *(_DWORD *)(a1 + 392);
      v76.m128i_i64[0] = v12;
      if ( v43 )
      {
        v44 = *(_QWORD *)(a1 + 376);
        v45 = 1;
        v46 = 0;
        v47 = (v43 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        result = v44 + 32LL * v47;
        v48 = *(_QWORD *)result;
        if ( v12 == *(_QWORD *)result )
        {
LABEL_56:
          v49 = *(__m128i **)(result + 16);
          v50 = (const __m128i **)(result + 8);
          if ( *(__m128i **)(result + 24) != v49 )
          {
            if ( v49 )
            {
              *v49 = _mm_loadu_si128(&v83);
              v49[1] = _mm_loadu_si128(&v84);
              v49[2].m128i_i64[0] = v85;
              v49 = *(__m128i **)(result + 16);
            }
            *(_QWORD *)(result + 16) = (char *)v49 + 40;
            return result;
          }
          return sub_1076F80(v50, v49, &v83);
        }
        while ( v48 != -4096 )
        {
          if ( !v46 && v48 == -8192 )
            v46 = (_QWORD *)result;
          v47 = (v43 - 1) & (v45 + v47);
          result = v44 + 32LL * v47;
          v48 = *(_QWORD *)result;
          if ( v12 == *(_QWORD *)result )
            goto LABEL_56;
          ++v45;
        }
        if ( !v46 )
          v46 = (_QWORD *)result;
        v54 = *(_DWORD *)(a1 + 384);
        ++*(_QWORD *)(a1 + 368);
        v55 = v54 + 1;
        v80[0].m128i_i64[0] = (__int64)v46;
        if ( 4 * (v54 + 1) < 3 * v43 )
        {
          if ( v43 - *(_DWORD *)(a1 + 388) - v55 > v43 >> 3 )
          {
LABEL_83:
            *(_DWORD *)(a1 + 384) = v55;
            if ( *v46 != -4096 )
              --*(_DWORD *)(a1 + 388);
            *v46 = v12;
            v49 = 0;
            v50 = (const __m128i **)(v46 + 1);
            *v50 = 0;
            v50[1] = 0;
            v50[2] = 0;
            return sub_1076F80(v50, v49, &v83);
          }
LABEL_95:
          sub_1077F20(a1 + 368, v43);
          sub_1076DD0(a1 + 368, v76.m128i_i64, v80);
          v12 = v76.m128i_i64[0];
          v46 = (_QWORD *)v80[0].m128i_i64[0];
          v55 = *(_DWORD *)(a1 + 384) + 1;
          goto LABEL_83;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 368);
        v80[0].m128i_i64[0] = 0;
      }
      v43 *= 2;
      goto LABEL_95;
    }
    v53 = *(__m128i **)(a1 + 128);
    if ( v53 == *(__m128i **)(a1 + 136) )
    {
      return sub_1076F80((const __m128i **)(a1 + 120), v53, &v83);
    }
    else
    {
      if ( v53 )
      {
        *v53 = _mm_loadu_si128(&v83);
        v53[1] = _mm_loadu_si128(&v84);
        result = v85;
        v53[2].m128i_i64[0] = v85;
        v53 = *(__m128i **)(a1 + 128);
      }
      *(_QWORD *)(a1 + 128) = (char *)v53 + 40;
    }
  }
  return result;
}
