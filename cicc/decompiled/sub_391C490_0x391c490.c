// Function: sub_391C490
// Address: 0x391c490
//
unsigned __int64 __fastcall sub_391C490(
        __int64 a1,
        __int64 a2,
        _QWORD *a3,
        __int64 a4,
        __int64 a5,
        _QWORD *a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v13; // rax
  __int64 v14; // rsi
  unsigned __int64 v15; // r13
  unsigned __int64 result; // rax
  __int64 v17; // r8
  __int64 v18; // r15
  __int64 v19; // rsi
  unsigned __int64 v20; // rcx
  _QWORD *v21; // rax
  unsigned __int64 v22; // rsi
  unsigned __int64 v23; // rax
  __int64 v24; // rcx
  __int32 v25; // eax
  _BYTE *v26; // rcx
  __int32 v27; // r14d
  __m128i *v28; // rsi
  __int64 v29; // r15
  unsigned __int64 v30; // rax
  __int64 v31; // rax
  _BYTE *v32; // rcx
  const char *v33; // rax
  unsigned __int64 v34; // rsi
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // rdi
  __int64 v37; // rax
  unsigned int v38; // esi
  __int64 v39; // r9
  unsigned int v40; // ecx
  __int64 v41; // rdx
  __m128i *v42; // rsi
  __m128i *v43; // rsi
  unsigned __int64 *v44; // rax
  __int64 v45; // rcx
  __int64 v46; // rdi
  unsigned int v47; // esi
  __int64 *v48; // rdx
  __int64 v49; // r8
  int v50; // r11d
  _QWORD *v51; // rdi
  int v52; // eax
  int v53; // edx
  unsigned __int64 *v54; // rdi
  int v55; // eax
  int v56; // ecx
  __int64 v57; // r8
  unsigned int v58; // eax
  __int64 v59; // rsi
  int v60; // r10d
  _QWORD *v61; // r9
  int v62; // eax
  int v63; // eax
  __int64 v64; // rsi
  _QWORD *v65; // r8
  unsigned int v66; // r12d
  int v67; // r9d
  __int64 v68; // rcx
  int v69; // edx
  int v70; // r10d
  __int64 v72; // [rsp+18h] [rbp-A8h]
  int v73; // [rsp+20h] [rbp-A0h]
  __int64 v74; // [rsp+20h] [rbp-A0h]
  _BYTE *v75; // [rsp+20h] [rbp-A0h]
  __int64 v76; // [rsp+20h] [rbp-A0h]
  __int64 v78; // [rsp+28h] [rbp-98h]
  _QWORD v79[2]; // [rsp+30h] [rbp-90h] BYREF
  _QWORD v80[2]; // [rsp+40h] [rbp-80h] BYREF
  __int16 v81; // [rsp+50h] [rbp-70h]
  __m128i v82; // [rsp+60h] [rbp-60h] BYREF
  __m128i v83; // [rsp+70h] [rbp-50h] BYREF
  unsigned __int64 v84; // [rsp+80h] [rbp-40h]

  v13 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a2 + 8) + 48LL))(
          *(_QWORD *)(a2 + 8),
          *(unsigned int *)(a5 + 12));
  v14 = a4;
  v15 = *(_QWORD *)(a4 + 24);
  v73 = *(_DWORD *)(v13 + 16);
  v78 = a9;
  v72 = *(unsigned int *)(a5 + 8) + sub_38D01B0((__int64)a3, v14);
  if ( *(_QWORD *)(v15 + 160) > 0xAu )
  {
    result = *(_QWORD *)(v15 + 152);
    if ( *(_QWORD *)result == 0x72615F74696E692ELL && *(_WORD *)(result + 8) == 24946 && *(_BYTE *)(result + 10) == 121 )
      return result;
  }
  if ( a8 )
  {
    v17 = *(_QWORD *)a2;
    if ( (v73 & 1) != 0 )
    {
      v83.m128i_i8[1] = 1;
      v33 = "No relocation available to represent this relative expression";
LABEL_37:
      v34 = *(_QWORD *)(a5 + 16);
      v82.m128i_i64[0] = (__int64)v33;
      v83.m128i_i8[0] = 3;
      return (unsigned __int64)sub_38BE3D0(v17, v34, (__int64)&v82);
    }
    v18 = *(_QWORD *)(a8 + 24);
    v19 = *(_QWORD *)v18;
    v20 = *(_QWORD *)v18 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v20 )
    {
      v23 = *(_QWORD *)v18 & 0xFFFFFFFFFFFFFFF8LL;
    }
    else
    {
      if ( (*(_BYTE *)(v18 + 9) & 0xC) != 8 )
        goto LABEL_7;
      *(_BYTE *)(v18 + 8) |= 4u;
      v76 = v17;
      v35 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v18 + 24));
      v17 = v76;
      v20 = 0;
      v36 = v35;
      v37 = v35 | *(_QWORD *)v18 & 7LL;
      *(_QWORD *)v18 = v37;
      LOBYTE(v19) = v37;
      if ( !v36 )
      {
LABEL_7:
        v21 = 0;
        if ( (v19 & 4) != 0 )
        {
          v44 = *(unsigned __int64 **)(v18 - 8);
          v20 = *v44;
          v21 = v44 + 2;
        }
        v79[0] = v21;
        v22 = *(_QWORD *)(a5 + 16);
        v83.m128i_i16[0] = 770;
        v80[0] = "symbol '";
        v80[1] = v79;
        v81 = 1283;
        v82.m128i_i64[0] = (__int64)v80;
        v79[1] = v20;
        v82.m128i_i64[1] = (__int64)"' can not be undefined in a subtraction expression";
        return (unsigned __int64)sub_38BE3D0(v17, v22, (__int64)&v82);
      }
      v23 = v37 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v23 )
      {
        if ( (*(_BYTE *)(v18 + 9) & 0xC) != 8
          || (*(_BYTE *)(v18 + 8) |= 4u,
              v23 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v18 + 24)),
              v17 = v76,
              *(_QWORD *)v18 = v23 | *(_QWORD *)v18 & 7LL,
              !v23) )
        {
LABEL_44:
          v83.m128i_i8[1] = 1;
          v33 = "Cannot represent a difference across sections";
          goto LABEL_37;
        }
      }
    }
    if ( v15 == *(_QWORD *)(v23 + 24) )
    {
      v78 = v72 + v78 - sub_38D0440(a3, v18);
      goto LABEL_13;
    }
    goto LABEL_44;
  }
LABEL_13:
  v24 = a7;
  if ( a7 )
  {
    v24 = *(_QWORD *)(a7 + 24);
    if ( v24 )
    {
      if ( (*(_BYTE *)(v24 + 9) & 0xC) == 8 )
        *(_BYTE *)(v24 + 8) |= 4u;
    }
  }
  v74 = v24;
  *a6 = 0;
  v25 = (*(__int64 (__fastcall **)(_QWORD, __int64 *, __int64))(**(_QWORD **)(a1 + 24) + 24LL))(
          *(_QWORD *)(a1 + 24),
          &a7,
          a5);
  v26 = (_BYTE *)v74;
  v27 = v25;
  if ( (unsigned int)(v25 - 8) > 1 )
  {
    if ( v25 == 6 )
      goto LABEL_19;
    goto LABEL_33;
  }
  if ( *(_BYTE *)(v15 + 148) )
    sub_16BD130("relocations for function or section offsets are only supported in metadata sections", 1u);
  v29 = v74;
  v30 = *(_QWORD *)v74 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v30 )
  {
    if ( (*(_BYTE *)(v74 + 9) & 0xC) != 8
      || (*(_BYTE *)(v74 + 8) |= 4u,
          v30 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v74 + 24)),
          *(_QWORD *)v74 = v30 | *(_QWORD *)v74 & 7LL,
          !v30) )
    {
      BUG();
    }
  }
  v31 = *(_QWORD *)(v30 + 24);
  if ( (unsigned __int8)(*(_BYTE *)(v31 + 148) - 1) <= 1u )
  {
    v45 = *(unsigned int *)(a1 + 304);
    v46 = *(_QWORD *)(a1 + 288);
    if ( (_DWORD)v45 )
    {
      v47 = (v45 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
      v48 = (__int64 *)(v46 + 16LL * v47);
      v49 = *v48;
      if ( v31 == *v48 )
      {
LABEL_61:
        v32 = (_BYTE *)v48[1];
        goto LABEL_31;
      }
      v69 = 1;
      while ( v49 != -8 )
      {
        v70 = v69 + 1;
        v47 = (v45 - 1) & (v69 + v47);
        v48 = (__int64 *)(v46 + 16LL * v47);
        v49 = *v48;
        if ( v31 == *v48 )
          goto LABEL_61;
        v69 = v70;
      }
    }
    v48 = (__int64 *)(v46 + 16 * v45);
    goto LABEL_61;
  }
  v32 = *(_BYTE **)(v31 + 8);
LABEL_31:
  if ( !v32 )
    sub_16BD130("section symbol is required for relocation", 1u);
  v75 = v32;
  v78 += sub_38D0440(a3, v29);
  v26 = v75;
LABEL_33:
  if ( (*v26 & 4) == 0 || !**((_QWORD **)v26 - 1) )
    sub_16BD130("relocations against un-named temporaries are not yet supported by wasm", 1u);
  v26[9] |= 2u;
LABEL_19:
  v82.m128i_i64[1] = (__int64)v26;
  v83.m128i_i32[2] = v27;
  v82.m128i_i64[0] = v72;
  v84 = v15;
  v83.m128i_i64[0] = v78;
  result = *(unsigned __int8 *)(v15 + 148);
  if ( (unsigned __int8)(result - 3) <= 7u || (unsigned __int8)(result - 13) <= 5u )
  {
    v28 = *(__m128i **)(a1 + 72);
    if ( v28 == *(__m128i **)(a1 + 80) )
      return sub_3919430((unsigned __int64 *)(a1 + 64), v28, &v82);
    if ( v28 )
    {
      *v28 = _mm_loadu_si128(&v82);
      v28[1] = _mm_loadu_si128(&v83);
      result = v84;
      v28[2].m128i_i64[0] = v84;
      v28 = *(__m128i **)(a1 + 72);
    }
    *(_QWORD *)(a1 + 72) = (char *)v28 + 40;
    return result;
  }
  result = (unsigned int)(result - 1);
  if ( (unsigned __int8)result > 1u )
  {
    v38 = *(_DWORD *)(a1 + 272);
    if ( v38 )
    {
      v39 = *(_QWORD *)(a1 + 256);
      v40 = (v38 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      result = v39 + 32LL * v40;
      v41 = *(_QWORD *)result;
      if ( v15 == *(_QWORD *)result )
      {
LABEL_48:
        v42 = *(__m128i **)(result + 16);
        if ( v42 != *(__m128i **)(result + 24) )
        {
          if ( v42 )
          {
            *v42 = _mm_loadu_si128(&v82);
            v42[1] = _mm_loadu_si128(&v83);
            v42[2].m128i_i64[0] = v84;
            v42 = *(__m128i **)(result + 16);
          }
          *(_QWORD *)(result + 16) = (char *)v42 + 40;
          return result;
        }
        v54 = (unsigned __int64 *)(result + 8);
        return sub_3919430(v54, v42, &v82);
      }
      v50 = 1;
      v51 = 0;
      while ( v41 != -8 )
      {
        if ( !v51 && v41 == -16 )
          v51 = (_QWORD *)result;
        v40 = (v38 - 1) & (v50 + v40);
        result = v39 + 32LL * v40;
        v41 = *(_QWORD *)result;
        if ( v15 == *(_QWORD *)result )
          goto LABEL_48;
        ++v50;
      }
      if ( !v51 )
        v51 = (_QWORD *)result;
      v52 = *(_DWORD *)(a1 + 264);
      ++*(_QWORD *)(a1 + 248);
      v53 = v52 + 1;
      if ( 4 * (v52 + 1) < 3 * v38 )
      {
        if ( v38 - *(_DWORD *)(a1 + 268) - v53 > v38 >> 3 )
        {
LABEL_70:
          *(_DWORD *)(a1 + 264) = v53;
          if ( *v51 != -8 )
            --*(_DWORD *)(a1 + 268);
          *v51 = v15;
          v42 = 0;
          v54 = v51 + 1;
          *v54 = 0;
          v54[1] = 0;
          v54[2] = 0;
          return sub_3919430(v54, v42, &v82);
        }
        sub_391A270(a1 + 248, v38);
        v62 = *(_DWORD *)(a1 + 272);
        if ( v62 )
        {
          v63 = v62 - 1;
          v64 = *(_QWORD *)(a1 + 256);
          v65 = 0;
          v66 = v63 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
          v67 = 1;
          v53 = *(_DWORD *)(a1 + 264) + 1;
          v51 = (_QWORD *)(v64 + 32LL * v66);
          v68 = *v51;
          if ( v15 != *v51 )
          {
            while ( v68 != -8 )
            {
              if ( v68 == -16 && !v65 )
                v65 = v51;
              v66 = v63 & (v67 + v66);
              v51 = (_QWORD *)(v64 + 32LL * v66);
              v68 = *v51;
              if ( v15 == *v51 )
                goto LABEL_70;
              ++v67;
            }
            if ( v65 )
              v51 = v65;
          }
          goto LABEL_70;
        }
LABEL_110:
        ++*(_DWORD *)(a1 + 264);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 248);
    }
    sub_391A270(a1 + 248, 2 * v38);
    v55 = *(_DWORD *)(a1 + 272);
    if ( v55 )
    {
      v56 = v55 - 1;
      v57 = *(_QWORD *)(a1 + 256);
      v58 = (v55 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v53 = *(_DWORD *)(a1 + 264) + 1;
      v51 = (_QWORD *)(v57 + 32LL * v58);
      v59 = *v51;
      if ( v15 != *v51 )
      {
        v60 = 1;
        v61 = 0;
        while ( v59 != -8 )
        {
          if ( v59 == -16 && !v61 )
            v61 = v51;
          v58 = v56 & (v60 + v58);
          v51 = (_QWORD *)(v57 + 32LL * v58);
          v59 = *v51;
          if ( v15 == *v51 )
            goto LABEL_70;
          ++v60;
        }
        if ( v61 )
          v51 = v61;
      }
      goto LABEL_70;
    }
    goto LABEL_110;
  }
  v43 = *(__m128i **)(a1 + 40);
  if ( v43 == *(__m128i **)(a1 + 48) )
    return sub_3919430((unsigned __int64 *)(a1 + 32), v43, &v82);
  if ( v43 )
  {
    *v43 = _mm_loadu_si128(&v82);
    v43[1] = _mm_loadu_si128(&v83);
    result = v84;
    v43[2].m128i_i64[0] = v84;
    v43 = *(__m128i **)(a1 + 40);
  }
  *(_QWORD *)(a1 + 40) = (char *)v43 + 40;
  return result;
}
