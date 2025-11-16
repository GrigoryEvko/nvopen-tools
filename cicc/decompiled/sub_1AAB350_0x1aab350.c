// Function: sub_1AAB350
// Address: 0x1aab350
//
__int64 __fastcall sub_1AAB350(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        char *a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        char a15)
{
  const char *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r14
  _QWORD *v20; // rax
  __int64 v21; // r13
  _QWORD *v22; // rax
  int v23; // eax
  __int64 v24; // rsi
  int v25; // ecx
  unsigned int v26; // edx
  __int64 *v27; // rax
  __int64 v28; // r8
  __int64 *v29; // rdi
  __int64 v30; // rdi
  __int64 *v31; // r14
  __int64 v32; // rsi
  unsigned __int8 *v33; // rsi
  __int64 v34; // rax
  __int64 v35; // rsi
  __m128 *v36; // rax
  __int64 v37; // rcx
  size_t v38; // r8
  unsigned __int8 *v39; // rdx
  __int64 v40; // rax
  double v41; // xmm4_8
  double v42; // xmm5_8
  _BYTE *v43; // rdi
  __int64 *v44; // r14
  __int64 *v45; // r14
  __int64 v46; // rdi
  unsigned __int64 v47; // rax
  __int64 v48; // r12
  unsigned __int64 v49; // rax
  __int64 v50; // r15
  unsigned __int64 v51; // rax
  unsigned __int64 v52; // rax
  __int64 i; // rbx
  __int64 v55; // rdx
  __int64 *v56; // rax
  unsigned __int64 v57; // rdx
  __int64 v58; // rdx
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r12
  __int64 v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // r8
  __int64 v65; // r9
  __int64 v66; // r14
  int v67; // eax
  __int64 v68; // rax
  int v69; // edx
  __int64 v70; // rax
  _QWORD *v71; // rdi
  int v72; // eax
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // r11
  unsigned int v77; // r9d
  int v78; // edi
  int v79; // r10d
  int v80; // edi
  __int64 v81; // [rsp+0h] [rbp-F0h]
  __int64 v82; // [rsp+8h] [rbp-E8h]
  size_t n; // [rsp+10h] [rbp-E0h]
  size_t na; // [rsp+10h] [rbp-E0h]
  size_t nb; // [rsp+10h] [rbp-E0h]
  __int64 v90; // [rsp+38h] [rbp-B8h]
  _QWORD v91[2]; // [rsp+40h] [rbp-B0h] BYREF
  _BYTE v92[16]; // [rsp+50h] [rbp-A0h] BYREF
  __m128 *v93; // [rsp+60h] [rbp-90h] BYREF
  __int64 v94; // [rsp+68h] [rbp-88h]
  __m128 v95; // [rsp+70h] [rbp-80h] BYREF
  unsigned __int8 *v96; // [rsp+80h] [rbp-70h] BYREF
  __m128 *v97; // [rsp+88h] [rbp-68h]
  _QWORD v98[3]; // [rsp+90h] [rbp-60h] BYREF
  int v99; // [rsp+ACh] [rbp-44h]

  if ( !sub_157F5F0(a1) )
    return 0;
  if ( !sub_157F790(a1) )
  {
    n = *(_QWORD *)(a1 + 56);
    v17 = sub_1649960(a1);
    v97 = (__m128 *)a4;
    v94 = v18;
    LOWORD(v98[0]) = 773;
    v93 = (__m128 *)v17;
    v96 = (unsigned __int8 *)&v93;
    v19 = sub_157E9C0(a1);
    v20 = (_QWORD *)sub_22077B0(64);
    v21 = (__int64)v20;
    if ( v20 )
      sub_157FB60(v20, v19, (__int64)&v96, n, a1);
    v22 = sub_1648A60(56, 1u);
    na = (size_t)v22;
    if ( v22 )
      sub_15F8590((__int64)v22, a1, v21);
    if ( a6 )
    {
      v23 = *(_DWORD *)(a6 + 24);
      if ( v23 )
      {
        v24 = *(_QWORD *)(a6 + 8);
        v25 = v23 - 1;
        v26 = (v23 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v27 = (__int64 *)(v24 + 16LL * v26);
        v28 = *v27;
        v29 = v27;
        if ( a1 == *v27 )
        {
LABEL_10:
          v30 = v29[1];
          if ( v30 && a1 == **(_QWORD **)(v30 + 32) )
          {
            if ( a1 == v28 )
            {
LABEL_13:
              v81 = v27[1];
              sub_13FD840(&v96, v81);
            }
            else
            {
              v72 = 1;
              while ( v28 != -8 )
              {
                v80 = v72 + 1;
                v26 = v25 & (v72 + v26);
                v27 = (__int64 *)(v24 + 16LL * v26);
                v28 = *v27;
                if ( a1 == *v27 )
                  goto LABEL_13;
                v72 = v80;
              }
              v81 = 0;
              sub_13FD840(&v96, 0);
            }
            v31 = (__int64 *)(na + 48);
            if ( (unsigned __int8 **)(na + 48) == &v96 )
            {
              if ( v96 )
                sub_161E7C0((__int64)&v96, (__int64)v96);
            }
            else
            {
              v32 = *(_QWORD *)(na + 48);
              if ( v32 )
                sub_161E7C0(na + 48, v32);
              v33 = v96;
              *(_QWORD *)(na + 48) = v96;
              if ( v33 )
                sub_1623210((__int64)&v96, v33, (__int64)v31);
            }
            v34 = sub_157EB90(a1);
            v35 = (__int64)&v93;
            v95.m128_i16[0] = 260;
            v93 = (__m128 *)(v34 + 240);
            sub_16E1010((__int64)&v96, (__int64)&v93);
            if ( v99 != 23
              || !*(_QWORD *)(sub_157ED60(a1) + 48)
              || (v73 = sub_157ED60(a1), v74 = sub_15C70A0(v73 + 48), *(_DWORD *)(v74 + 8) != 2)
              || !*(_QWORD *)(v74 - 8)
              || !*(_QWORD *)(na + 48)
              || (v75 = sub_15C70A0((__int64)v31), *(_DWORD *)(v75 + 8) == 2) && *(_QWORD *)(v75 - 8) )
            {
              if ( v96 != (unsigned __int8 *)v98 )
              {
                v35 = v98[0] + 1LL;
                j_j___libc_free_0(v96, v98[0] + 1LL);
              }
              goto LABEL_22;
            }
            if ( v96 != (unsigned __int8 *)v98 )
              j_j___libc_free_0(v96, v98[0] + 1LL);
            v35 = *(_QWORD *)(sub_157ED60(a1) + 48);
            v96 = (unsigned __int8 *)v35;
            if ( v35 )
            {
              sub_1623A60((__int64)&v96, v35, 2);
              if ( v31 == (__int64 *)&v96 )
              {
                v35 = (__int64)v96;
                if ( v96 )
                  sub_161E7C0((__int64)&v96, (__int64)v96);
                goto LABEL_22;
              }
              v35 = *(_QWORD *)(na + 48);
              if ( !v35 )
              {
LABEL_111:
                v35 = (__int64)v96;
                *(_QWORD *)(na + 48) = v96;
                if ( v35 )
                  sub_1623210((__int64)&v96, (unsigned __int8 *)v35, (__int64)v31);
                goto LABEL_22;
              }
            }
            else if ( v31 == (__int64 *)&v96 || (v35 = *(_QWORD *)(na + 48)) == 0 )
            {
LABEL_22:
              v82 = sub_13FCB50(v81);
              goto LABEL_41;
            }
            sub_161E7C0((__int64)v31, v35);
            goto LABEL_111;
          }
        }
        else
        {
          v76 = *v27;
          v77 = v26;
          v78 = 1;
          while ( v76 != -8 )
          {
            v79 = v78 + 1;
            v77 = v25 & (v78 + v77);
            v29 = (__int64 *)(v24 + 16LL * v77);
            v76 = *v29;
            if ( a1 == *v29 )
              goto LABEL_10;
            v78 = v79;
          }
        }
      }
    }
    v35 = *(_QWORD *)(sub_157ED60(a1) + 48);
    v96 = (unsigned __int8 *)v35;
    v44 = (__int64 *)(na + 48);
    if ( v35 )
    {
      sub_1623A60((__int64)&v96, v35, 2);
      if ( v44 == (__int64 *)&v96 )
      {
        v35 = (__int64)v96;
        if ( v96 )
          sub_161E7C0((__int64)&v96, (__int64)v96);
        goto LABEL_40;
      }
      v35 = *(_QWORD *)(na + 48);
      if ( !v35 )
      {
LABEL_55:
        v35 = (__int64)v96;
        *(_QWORD *)(na + 48) = v96;
        if ( v35 )
          sub_1623210((__int64)&v96, (unsigned __int8 *)v35, (__int64)v44);
        goto LABEL_40;
      }
    }
    else if ( v44 == (__int64 *)&v96 || (v35 = *(_QWORD *)(na + 48)) == 0 )
    {
LABEL_40:
      v82 = 0;
      v81 = 0;
LABEL_41:
      if ( (_DWORD)a3 )
      {
        v45 = a2;
        do
        {
          v46 = *v45++;
          v47 = sub_157EBA0(v46);
          sub_1648780(v47, a1, v21);
        }
        while ( v45 != &a2[(unsigned int)(a3 - 1) + 1] );
      }
      else if ( !a3 )
      {
        v90 = a1;
        for ( i = *(_QWORD *)(a1 + 48); ; i = *(_QWORD *)(i + 8) )
        {
          if ( !i )
            BUG();
          if ( *(_BYTE *)(i - 8) != 77 )
            break;
          v61 = i - 24;
          v66 = sub_1599EF0(*(__int64 ***)(i - 24));
          v67 = *(_DWORD *)(i - 4) & 0xFFFFFFF;
          if ( v67 == *(_DWORD *)(i + 32) )
          {
            sub_15F55D0(i - 24, v35, v62, v63, v64, v65);
            v67 = *(_DWORD *)(i - 4) & 0xFFFFFFF;
          }
          v68 = (v67 + 1) & 0xFFFFFFF;
          v35 = (unsigned int)(v68 - 1);
          v69 = v68 | *(_DWORD *)(i - 4) & 0xF0000000;
          *(_DWORD *)(i - 4) = v69;
          if ( (v69 & 0x40000000) != 0 )
            v55 = *(_QWORD *)(i - 32);
          else
            v55 = v61 - 24 * v68;
          v56 = (__int64 *)(v55 + 24LL * (unsigned int)v35);
          if ( *v56 )
          {
            v35 = v56[1];
            v57 = v56[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v57 = v35;
            if ( v35 )
              *(_QWORD *)(v35 + 16) = *(_QWORD *)(v35 + 16) & 3LL | v57;
          }
          *v56 = v66;
          if ( v66 )
          {
            v58 = *(_QWORD *)(v66 + 8);
            v56[1] = v58;
            if ( v58 )
            {
              v35 = (unsigned __int64)(v56 + 1) | *(_QWORD *)(v58 + 16) & 3LL;
              *(_QWORD *)(v58 + 16) = v35;
            }
            v56[2] = (v66 + 8) | v56[2] & 3;
            *(_QWORD *)(v66 + 8) = v56;
          }
          v59 = *(_DWORD *)(i - 4) & 0xFFFFFFF;
          if ( (*(_BYTE *)(i - 1) & 0x40) != 0 )
            v60 = *(_QWORD *)(i - 32);
          else
            v60 = v61 - 24 * v59;
          *(_QWORD *)(v60 + 8LL * (unsigned int)(v59 - 1) + 24LL * *(unsigned int *)(i + 32) + 8) = v21;
        }
        LOBYTE(v96) = 0;
        sub_1AA9AF0(v90, v21, a2, 0, a5, a6, a15, (__int64 *)&v96);
        goto LABEL_45;
      }
      LOBYTE(v96) = 0;
      sub_1AA9AF0(a1, v21, a2, a3, a5, a6, a15, (__int64 *)&v96);
      sub_1AA5740(a1, v21, a2, a3, na, (char)v96);
LABEL_45:
      if ( v82 )
      {
        v48 = sub_13FCB50(v81);
        if ( v82 != v48 )
        {
          v49 = sub_157EBA0(v82);
          v50 = *(_QWORD *)(v49 + 48);
          if ( v50 || *(__int16 *)(v49 + 18) < 0 )
            v50 = sub_1625940(v49, "llvm.loop", 9u);
          v51 = sub_157EBA0(v48);
          sub_1626100(v51, "llvm.loop", 9u, v50);
          v52 = sub_157EBA0(v82);
          sub_1626100(v52, "llvm.loop", 9u, 0);
        }
      }
      return v21;
    }
    sub_161E7C0((__int64)v44, v35);
    goto LABEL_55;
  }
  v91[0] = v92;
  v91[1] = 0x200000000LL;
  v96 = (unsigned __int8 *)v98;
  if ( !a4 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v36 = (__m128 *)strlen(a4);
  v93 = v36;
  v38 = (size_t)v36;
  if ( (unsigned __int64)v36 > 0xF )
  {
    nb = (size_t)v36;
    v70 = sub_22409D0(&v96, &v93, 0);
    v38 = nb;
    v96 = (unsigned __int8 *)v70;
    v71 = (_QWORD *)v70;
    v98[0] = v93;
    goto LABEL_81;
  }
  if ( v36 != (__m128 *)1 )
  {
    if ( !v36 )
    {
      v39 = (unsigned __int8 *)v98;
      goto LABEL_27;
    }
    v71 = v98;
LABEL_81:
    memcpy(v71, a4, v38);
    v36 = v93;
    v39 = v96;
    goto LABEL_27;
  }
  LOBYTE(v98[0]) = *a4;
  v39 = (unsigned __int8 *)v98;
LABEL_27:
  v97 = v36;
  v36->m128_i8[(_QWORD)v39] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - (_QWORD)v97) <= 8 )
    sub_4262D8((__int64)"basic_string::append");
  v40 = sub_2241490(&v96, ".split-lp", 9, v37, v38);
  v93 = &v95;
  if ( *(_QWORD *)v40 == v40 + 16 )
  {
    a7 = (__m128)_mm_loadu_si128((const __m128i *)(v40 + 16));
    v95 = a7;
  }
  else
  {
    v93 = *(__m128 **)v40;
    v95.m128_u64[0] = *(_QWORD *)(v40 + 16);
  }
  v94 = *(_QWORD *)(v40 + 8);
  *(_QWORD *)v40 = v40 + 16;
  *(_QWORD *)(v40 + 8) = 0;
  *(_BYTE *)(v40 + 16) = 0;
  if ( v96 != (unsigned __int8 *)v98 )
    j_j___libc_free_0(v96, v98[0] + 1LL);
  sub_1AAA850(a1, a2, a3, a4, v93, (__int64)v91, a7, a8, a9, a10, v41, v42, a13, a14, a5, a6, a15);
  v43 = (_BYTE *)v91[0];
  v21 = *(_QWORD *)v91[0];
  if ( v93 != &v95 )
  {
    j_j___libc_free_0(v93, v95.m128_u64[0] + 1);
    v43 = (_BYTE *)v91[0];
  }
  if ( v43 != v92 )
    _libc_free((unsigned __int64)v43);
  return v21;
}
