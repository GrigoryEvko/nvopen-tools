// Function: sub_195BF20
// Address: 0x195bf20
//
__int64 __fastcall sub_195BF20(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned int v10; // r14d
  __int64 v14; // rbx
  __int64 v15; // rax
  double v16; // xmm4_8
  double v17; // xmm5_8
  __int64 v18; // r15
  unsigned __int64 v19; // rdi
  int v20; // eax
  unsigned __int64 v21; // r15
  char v22; // al
  __int64 v23; // r14
  __int64 v24; // rax
  __int64 v25; // rbx
  __int64 v26; // rax
  _BYTE *v27; // rax
  __int64 v28; // rax
  double v29; // xmm4_8
  double v30; // xmm5_8
  __int64 v31; // rbx
  __int64 v32; // rax
  __int64 *v33; // rax
  unsigned __int64 v34; // rax
  __int64 v35; // rbx
  int v36; // eax
  int v37; // ebx
  __int64 v38; // r15
  _QWORD *v39; // rax
  _QWORD *v40; // r9
  __int64 v41; // rax
  __int64 *v42; // rax
  char v43; // al
  _QWORD *v44; // rax
  __int64 v45; // rax
  unsigned int v46; // r15d
  _QWORD *v47; // r14
  unsigned int v48; // eax
  __int64 v49; // r12
  _QWORD *v50; // rdi
  double v51; // xmm4_8
  double v52; // xmm5_8
  __int64 v53; // rbx
  unsigned __int64 v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rbx
  bool v57; // al
  unsigned __int64 v58; // rsi
  __int64 v59; // rdx
  __int64 i; // rbx
  __int64 v61; // rax
  unsigned __int64 v62; // rax
  __int64 v63; // rax
  double v64; // xmm4_8
  double v65; // xmm5_8
  __int64 v66; // rdi
  __int64 v67; // rsi
  _QWORD *v68; // rdx
  __int64 *v69; // rax
  __int64 v70; // rdx
  __int64 v71; // rcx
  __int64 *v72; // rsi
  unsigned int v73; // edi
  __int64 *v74; // rdx
  __int64 v75; // [rsp+0h] [rbp-B0h]
  unsigned int v76; // [rsp+8h] [rbp-A8h]
  __int64 v77; // [rsp+10h] [rbp-A0h]
  int v78; // [rsp+18h] [rbp-98h]
  __int64 v79; // [rsp+18h] [rbp-98h]
  __int64 v80; // [rsp+20h] [rbp-90h]
  __int64 v81; // [rsp+20h] [rbp-90h]
  unsigned int v82; // [rsp+20h] [rbp-90h]
  int v83; // [rsp+20h] [rbp-90h]
  __int64 v84; // [rsp+20h] [rbp-90h]
  int v85; // [rsp+28h] [rbp-88h]
  __int64 v86; // [rsp+28h] [rbp-88h]
  __int64 v87; // [rsp+28h] [rbp-88h]
  unsigned __int64 v88; // [rsp+30h] [rbp-80h] BYREF
  unsigned __int64 v89; // [rsp+38h] [rbp-78h] BYREF
  __int64 v90; // [rsp+40h] [rbp-70h] BYREF
  __int64 v91; // [rsp+48h] [rbp-68h] BYREF
  __m128i v92; // [rsp+50h] [rbp-60h] BYREF
  __m128i *v93; // [rsp+60h] [rbp-50h] BYREF
  __int64 v94; // [rsp+68h] [rbp-48h]
  _QWORD v95[8]; // [rsp+70h] [rbp-40h] BYREF

  if ( sub_15CD6A0(*(_QWORD *)(a1 + 24), a2) )
    return 0;
  v14 = *(_QWORD *)(a2 + 8);
  if ( v14 )
  {
    while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v14) + 16) - 25) > 9u )
    {
      v14 = *(_QWORD *)(v14 + 8);
      if ( !v14 )
        goto LABEL_29;
    }
  }
  else
  {
LABEL_29:
    v32 = *(_QWORD *)(*(_QWORD *)(a2 + 56) + 80LL);
    if ( !v32 || a2 != v32 - 24 )
      return 0;
  }
  v15 = sub_157F0B0(a2);
  v18 = v15;
  if ( !v15 )
  {
LABEL_10:
    if ( byte_4FB0380 )
      goto LABEL_11;
    goto LABEL_26;
  }
  v19 = sub_157EBA0(v15);
  v20 = *(unsigned __int8 *)(v19 + 16);
  if ( (unsigned int)(v20 - 24) <= 6 )
  {
    if ( (unsigned int)(v20 - 24) > 4 )
      goto LABEL_10;
    goto LABEL_9;
  }
  if ( (unsigned int)(v20 - 32) > 2 )
  {
LABEL_9:
    LOBYTE(v10) = (unsigned int)sub_15F4D60(v19) == 1 && a2 != v18;
    if ( (_BYTE)v10 )
    {
      if ( !*(_WORD *)(a2 + 18) || (v53 = sub_159BF40(a2), sub_159D9E0(v53), !*(_QWORD *)(v53 + 8)) )
      {
        v44 = *(_QWORD **)(a1 + 64);
        if ( *(_QWORD **)(a1 + 72) == v44 )
        {
          v68 = &v44[*(unsigned int *)(a1 + 84)];
          if ( v44 == v68 )
          {
LABEL_142:
            v44 = v68;
          }
          else
          {
            while ( v18 != *v44 )
            {
              if ( v68 == ++v44 )
                goto LABEL_142;
            }
          }
        }
        else
        {
          v44 = sub_16CC9F0(a1 + 56, v18);
          if ( v18 == *v44 )
          {
            v70 = *(_QWORD *)(a1 + 72);
            if ( v70 == *(_QWORD *)(a1 + 64) )
              v71 = *(unsigned int *)(a1 + 84);
            else
              v71 = *(unsigned int *)(a1 + 80);
            v68 = (_QWORD *)(v70 + 8 * v71);
          }
          else
          {
            v45 = *(_QWORD *)(a1 + 72);
            if ( v45 != *(_QWORD *)(a1 + 64) )
              goto LABEL_73;
            v44 = (_QWORD *)(v45 + 8LL * *(unsigned int *)(a1 + 84));
            v68 = v44;
          }
        }
        if ( v68 != v44 )
        {
          *v44 = -2;
          v69 = *(__int64 **)(a1 + 64);
          ++*(_DWORD *)(a1 + 88);
          if ( *(__int64 **)(a1 + 72) != v69 )
          {
LABEL_134:
            sub_16CCBA0(a1 + 56, a2);
            goto LABEL_73;
          }
          v72 = &v69[*(unsigned int *)(a1 + 84)];
          v73 = *(_DWORD *)(a1 + 84);
          if ( v69 == v72 )
          {
LABEL_154:
            if ( v73 >= *(_DWORD *)(a1 + 80) )
              goto LABEL_134;
            *(_DWORD *)(a1 + 84) = v73 + 1;
            *v72 = a2;
            ++*(_QWORD *)(a1 + 56);
          }
          else
          {
            v74 = 0;
            while ( a2 != *v69 )
            {
              if ( *v69 == -2 )
                v74 = v69;
              if ( v72 == ++v69 )
              {
                if ( !v74 )
                  goto LABEL_154;
                *v74 = a2;
                --*(_DWORD *)(a1 + 88);
                ++*(_QWORD *)(a1 + 56);
                break;
              }
            }
          }
        }
LABEL_73:
        sub_13EB690(*(__int64 **)(a1 + 8), v18);
        sub_1AF1D40(a2, 0, *(_QWORD *)(a1 + 24));
        if ( !(unsigned __int8)sub_14AE980(a2) )
        {
          sub_13EB690(*(__int64 **)(a1 + 8), a2);
          return v10;
        }
        return 1;
      }
    }
    goto LABEL_10;
  }
  if ( byte_4FB0380 )
    goto LABEL_11;
LABEL_26:
  if ( (unsigned __int8)sub_1954210(a1, a2, a3, a4, a5, a6, v16, v17, a9, a10) )
    return 1;
LABEL_11:
  if ( *(_BYTE *)(a1 + 49) && (unsigned __int8)sub_195BDE0(a1, a2, a3, a4, a5, a6, v16, v17, a9, a10) )
    return 1;
  v21 = sub_157EBA0(a2);
  v22 = *(_BYTE *)(v21 + 16);
  switch ( v22 )
  {
    case 26:
      if ( (*(_DWORD *)(v21 + 20) & 0xFFFFFFF) != 1 )
      {
        v85 = 0;
        v23 = *(_QWORD *)(v21 - 72);
        break;
      }
      return 0;
    case 27:
      if ( (*(_BYTE *)(v21 + 23) & 0x40) != 0 )
        v33 = *(__int64 **)(v21 - 8);
      else
        v33 = (__int64 *)(v21 - 24LL * (*(_DWORD *)(v21 + 20) & 0xFFFFFFF));
      v85 = 0;
      v23 = *v33;
      if ( !*v33 )
        goto LABEL_36;
      break;
    case 28:
      v41 = *(_DWORD *)(v21 + 20) & 0xFFFFFFF;
      if ( (_DWORD)v41 == 1 )
        return 0;
      if ( (*(_BYTE *)(v21 + 23) & 0x40) != 0 )
        v42 = *(__int64 **)(v21 - 8);
      else
        v42 = (__int64 *)(v21 - 24 * v41);
      v85 = 1;
      v23 = sub_1649C60(*v42);
      break;
    default:
      return 0;
  }
  LOBYTE(v24) = *(_BYTE *)(v23 + 16);
  if ( (unsigned __int8)v24 > 0x17u )
  {
    v25 = *(_QWORD *)a1;
    v26 = sub_157EB90(a2);
    v27 = (_BYTE *)sub_1632FA0(v26);
    v28 = sub_14DD210((__int64 *)v23, v27, v25);
    v31 = v28;
    if ( v28 )
    {
      sub_164D160(v23, v28, a3, a4, a5, a6, v29, v30, a9, a10);
      if ( (unsigned __int8)sub_1AE9990(v23, *(_QWORD *)a1) )
        sub_15F20C0((_QWORD *)v23);
      LOBYTE(v24) = *(_BYTE *)(v31 + 16);
      v23 = v31;
    }
    else
    {
      v24 = *(unsigned __int8 *)(v23 + 16);
    }
  }
  if ( (_BYTE)v24 == 9 )
  {
    v46 = 0;
    v93 = 0;
    v82 = sub_1952280(a2);
    v94 = 0;
    v95[0] = 0;
    v47 = (_QWORD *)sub_157EBA0(a2);
    v48 = sub_15F4D60((__int64)v47);
    sub_1953AE0((const __m128i **)&v93, v48);
    v78 = sub_15F4D60((__int64)v47);
    if ( v78 )
    {
      do
      {
        if ( v82 != v46 )
        {
          v87 = sub_15F4DF0((__int64)v47, v46);
          sub_157F2D0(v87, a2, 1);
          v92.m128i_i64[0] = a2;
          v92.m128i_i64[1] = v87 | 4;
          sub_19541D0((__int64)&v93, &v92);
        }
        ++v46;
      }
      while ( v78 != v46 );
    }
    v49 = sub_15F4DF0((__int64)v47, v82);
    v50 = sub_1648A60(56, 1u);
    if ( v50 )
      sub_15F8320((__int64)v50, v49, (__int64)v47);
    sub_15F20C0(v47);
    sub_15CD9D0(*(_QWORD *)(a1 + 24), v93->m128i_i64, (v94 - (__int64)v93) >> 4);
    if ( v93 )
      j_j___libc_free_0(v93, v95[0] - (_QWORD)v93);
    return 1;
  }
  if ( v85 == 1 )
  {
    if ( *(_BYTE *)(sub_1649C60(v23) + 16) == 4 )
      goto LABEL_23;
    LOBYTE(v24) = *(_BYTE *)(v23 + 16);
  }
  else if ( (_BYTE)v24 == 13 )
  {
LABEL_23:
    v10 = 1;
    sub_1AEE9C0(a2, 1, 0, *(_QWORD *)(a1 + 24));
    return v10;
  }
  if ( (unsigned __int8)v24 <= 0x17u )
    return sub_1959640(a1, v23, a2);
  if ( (unsigned __int8)(v24 - 75) > 1u )
    goto LABEL_63;
  v34 = sub_157EBA0(a2);
  if ( *(_BYTE *)(v34 + 16) != 26 )
  {
    if ( *(_QWORD *)(v23 - 24) )
      goto LABEL_65;
LABEL_36:
    BUG();
  }
  v35 = *(_QWORD *)(v23 - 24);
  v80 = v34;
  if ( !v35 )
    goto LABEL_36;
  if ( *(_BYTE *)(v35 + 16) > 0x10u )
    goto LABEL_65;
  if ( sub_15CD740(*(_QWORD *)(a1 + 24)) )
    sub_13EBC00(*(__int64 **)(a1 + 8));
  else
    sub_13EBC50(*(__int64 **)(a1 + 8));
  v36 = sub_13F3450(*(__int64 **)(a1 + 8), *(_WORD *)(v23 + 18) & 0x7FFF, *(_QWORD *)(v23 - 48), v35, v80);
  v37 = v36;
  if ( v36 != -1 )
  {
    v86 = v80;
    v38 = *(_QWORD *)(v80 + -24 - 24LL * (v36 == 1));
    sub_157F2D0(v38, a2, 1);
    v81 = *(_QWORD *)(v80 + -24 - 24LL * (v37 != 1));
    v39 = sub_1648A60(56, 1u);
    v40 = (_QWORD *)v86;
    if ( v39 )
    {
      sub_15F8320((__int64)v39, v81, v86);
      v40 = (_QWORD *)v86;
    }
    sub_15F20C0(v40);
    if ( *(_QWORD *)(v23 + 8) )
    {
      if ( a2 == *(_QWORD *)(v23 + 40) )
      {
        v66 = *(_QWORD *)v23;
        if ( v37 == 1 )
          v67 = sub_15A0600(v66);
        else
          v67 = sub_15A0640(v66);
        sub_1952070((_QWORD *)v23, v67);
      }
    }
    else
    {
      sub_15F20C0((_QWORD *)v23);
    }
    sub_15CDBF0(*(_QWORD *)(a1 + 24), a2, v38);
    return 1;
  }
  if ( (unsigned __int8)sub_1953430(a1, v23, a2) )
    return 1;
  LOBYTE(v24) = *(_BYTE *)(v23 + 16);
LABEL_63:
  if ( (_BYTE)v24 == 77 && a2 == *(_QWORD *)(v23 + 40) )
  {
    v54 = sub_157EBA0(a2);
    if ( *(_BYTE *)(v54 + 16) == 26
      && (unsigned __int8)sub_1625AE0(v54, &v88, &v89)
      && (*(_DWORD *)(v23 + 20) & 0xFFFFFFF) != 0 )
    {
      v79 = 0;
      v77 = 8LL * (*(_DWORD *)(v23 + 20) & 0xFFFFFFF);
      do
      {
        if ( (*(_BYTE *)(v23 + 23) & 0x40) != 0 )
          v55 = *(_QWORD *)(v23 - 8);
        else
          v55 = v23 - 24LL * (*(_DWORD *)(v23 + 20) & 0xFFFFFFF);
        v56 = *(_QWORD *)(v55 + 3 * v79);
        if ( *(_BYTE *)(v56 + 16) == 13 && sub_1642F90(*(_QWORD *)v56, 1) )
        {
          if ( *(_DWORD *)(v56 + 32) <= 0x40u )
          {
            v57 = *(_QWORD *)(v56 + 24) == 1;
          }
          else
          {
            v83 = *(_DWORD *)(v56 + 32);
            v57 = v83 - 1 == (unsigned int)sub_16A57B0(v56 + 24);
          }
          v58 = v88 + v89;
          if ( v57 )
            v76 = sub_16AF730(v88, v58);
          else
            v76 = sub_16AF730(v89, v58);
          if ( (*(_BYTE *)(v23 + 23) & 0x40) != 0 )
            v59 = *(_QWORD *)(v23 - 8);
          else
            v59 = v23 - 24LL * (*(_DWORD *)(v23 + 20) & 0xFFFFFFF);
          v84 = a2;
          for ( i = *(_QWORD *)(v79 + v59 + 24LL * *(unsigned int *)(v23 + 56) + 8); ; i = v61 )
          {
            v62 = sub_157EBA0(i);
            if ( *(_BYTE *)(v62 + 16) == 26 && (*(_DWORD *)(v62 + 20) & 0xFFFFFFF) == 3 )
              break;
            v61 = sub_157F0B0(i);
            v84 = i;
            if ( !v61 )
              goto LABEL_65;
          }
          if ( !i )
            break;
          v75 = v62;
          if ( !(unsigned __int8)sub_1625AE0(v62, &v90, &v91) )
          {
            sub_16AF710(&v93, 0x32u, 0x64u);
            if ( (unsigned int)v93 > v76 )
            {
              v94 = 0x200000000LL;
              v93 = (__m128i *)v95;
              if ( *(_QWORD *)(v75 - 24) == v84 )
              {
                HIDWORD(v95[0]) = 0x80000000 - v76;
                LODWORD(v94) = 2;
                LODWORD(v95[0]) = v76;
              }
              else
              {
                LODWORD(v95[0]) = 0x80000000 - v76;
                LODWORD(v94) = 2;
                HIDWORD(v95[0]) = v76;
              }
              v92.m128i_i64[0] = sub_157E9C0(*(_QWORD *)(v75 + 40));
              v63 = sub_161BD30(&v92, (unsigned int *)v93, (unsigned int)v94);
              sub_1625C10(v75, 2, v63);
              if ( v93 != (__m128i *)v95 )
                _libc_free((unsigned __int64)v93);
            }
          }
        }
        v79 += 8;
      }
      while ( v77 != v79 );
    }
  }
LABEL_65:
  if ( (unsigned __int8)sub_1959640(a1, v23, a2) )
    return 1;
  v43 = *(_BYTE *)(v23 + 16);
  if ( v43 != 77 )
  {
    if ( v43 == 52 && a2 == *(_QWORD *)(v23 + 40) && *(_BYTE *)(sub_157EBA0(a2) + 16) == 26 )
      return sub_195A8F0(a1, (__int64 ***)v23, a3, a4, a5, a6, v51, v52, a9, a10);
    return sub_19531C0(a1, a2);
  }
  if ( a2 != *(_QWORD *)(v23 + 40) || *(_BYTE *)(sub_157EBA0(a2) + 16) != 26 )
    return sub_19531C0(a1, a2);
  return sub_195A7D0(a1, v23, a3, a4, a5, a6, v64, v65, a9, a10);
}
