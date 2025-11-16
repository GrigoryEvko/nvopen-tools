// Function: sub_278D170
// Address: 0x278d170
//
__int64 __fastcall sub_278D170(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v5; // r15d
  __int64 v7; // r12
  unsigned __int8 *v9; // r13
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  bool v13; // bl
  __int64 v14; // rax
  unsigned __int8 v17; // al
  __int64 v18; // r13
  __int64 v19; // r15
  __int64 v20; // r15
  unsigned __int8 *v21; // r13
  __int64 v22; // rbx
  unsigned __int64 *v23; // rbx
  unsigned __int64 *v24; // r13
  unsigned __int64 v25; // rdi
  __int64 v26; // rax
  char v27; // al
  __int64 v28; // r8
  int v29; // eax
  int v30; // ecx
  __m128i v31; // rax
  __int64 v32; // r15
  __m128i v33; // rax
  unsigned __int64 v34; // rax
  __int64 v35; // rax
  bool v36; // bl
  __int64 v37; // rax
  int v38; // eax
  __int64 v39; // r13
  int v40; // eax
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rbx
  __int64 v44; // r14
  unsigned __int8 *v45; // r12
  __int64 v46; // r15
  __int64 v47; // rax
  __int64 v48; // rsi
  __m128i v49; // xmm1
  __m128i v50; // xmm2
  _QWORD *v51; // rcx
  unsigned __int64 v52; // rax
  __int64 v53; // rax
  __m128i v54; // xmm4
  __m128i v55; // xmm5
  _QWORD *v56; // rcx
  __int64 v57; // rsi
  unsigned __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rcx
  __int64 v61; // rsi
  unsigned int v62; // edx
  __int64 *v63; // rax
  __int64 v64; // rdi
  __int64 v65; // [rsp+8h] [rbp-268h]
  unsigned __int8 *v66; // [rsp+18h] [rbp-258h]
  __int64 v67; // [rsp+20h] [rbp-250h]
  bool v68; // [rsp+28h] [rbp-248h]
  __int64 *v69; // [rsp+28h] [rbp-248h]
  bool v70; // [rsp+28h] [rbp-248h]
  __int64 v71; // [rsp+28h] [rbp-248h]
  int v72; // [rsp+28h] [rbp-248h]
  __int64 v73; // [rsp+30h] [rbp-240h]
  unsigned __int64 v74; // [rsp+38h] [rbp-238h]
  unsigned __int64 v75; // [rsp+38h] [rbp-238h]
  __m128i v76; // [rsp+40h] [rbp-230h] BYREF
  __m128i v77; // [rsp+50h] [rbp-220h] BYREF
  __m128i v78; // [rsp+60h] [rbp-210h] BYREF
  _QWORD v79[4]; // [rsp+70h] [rbp-200h] BYREF
  __m128i v80; // [rsp+90h] [rbp-1E0h] BYREF
  __m128i v81; // [rsp+A0h] [rbp-1D0h]
  __m128i v82; // [rsp+B0h] [rbp-1C0h]
  unsigned __int64 *v83; // [rsp+E0h] [rbp-190h]
  unsigned int v84; // [rsp+E8h] [rbp-188h]
  char v85; // [rsp+F0h] [rbp-180h] BYREF

  v5 = a4 & 7;
  v7 = a1;
  if ( v5 == 1 )
  {
    v74 = a4 & 0xFFFFFFFFFFFFFFF8LL;
    v73 = sub_B43CC0(a3);
    v17 = *(_BYTE *)v74;
    if ( *(_BYTE *)v74 == 62 )
    {
      if ( !a5 )
        goto LABEL_27;
      v68 = sub_B46500((unsigned __int8 *)a3);
      if ( (unsigned __int8)v68 > (unsigned __int8)sub_B46500((unsigned __int8 *)v74) )
        goto LABEL_27;
      v40 = sub_2A88FC0(*(_QWORD *)(a3 + 8), a5, v74, v73);
      if ( v40 != -1 )
      {
        *(_DWORD *)(a1 + 8) = 0;
        v41 = *(_QWORD *)(v74 - 64);
        *(_DWORD *)(a1 + 12) = v40;
        *(_QWORD *)(a1 + 16) = 0;
        *(_QWORD *)a1 = v41;
        *(_QWORD *)(a1 + 24) = 0;
        *(_BYTE *)(a1 + 32) = 1;
        return v7;
      }
      v17 = *(_BYTE *)v74;
    }
    if ( v17 != 61 )
    {
LABEL_73:
      if ( v17 == 85 )
      {
        v37 = *(_QWORD *)(v74 - 32);
        if ( v37 )
        {
          if ( !*(_BYTE *)v37
            && *(_QWORD *)(v37 + 24) == *(_QWORD *)(v74 + 80)
            && (*(_BYTE *)(v37 + 33) & 0x20) != 0
            && (unsigned int)(*(_DWORD *)(v37 + 36) - 238) <= 7
            && ((1LL << (*(_BYTE *)(v37 + 36) + 18)) & 0xAD) != 0 )
          {
            if ( a5 )
            {
              if ( !sub_B46500((unsigned __int8 *)a3) )
              {
                v38 = sub_2A89100(*(_QWORD *)(a3 + 8), a5, v74, v73);
                if ( v38 != -1 )
                {
                  *(_QWORD *)v7 = v74;
                  *(_DWORD *)(v7 + 8) = 2;
                  *(_DWORD *)(v7 + 12) = v38;
                  *(_QWORD *)(v7 + 16) = 0;
                  *(_QWORD *)(v7 + 24) = 0;
                  *(_BYTE *)(v7 + 32) = 1;
                  return v7;
                }
              }
            }
          }
        }
      }
LABEL_27:
      v18 = sub_B2BE50(**(_QWORD **)(a2 + 96));
      if ( !sub_B6EA50(v18) )
      {
        v39 = sub_B6F970(v18);
        if ( !(*(unsigned __int8 (__fastcall **)(__int64, char *, __int64))(*(_QWORD *)v39 + 32LL))(v39, "gvn", 3)
          && !(*(unsigned __int8 (__fastcall **)(__int64, char *, __int64))(*(_QWORD *)v39 + 40LL))(v39, "gvn", 3)
          && !(*(unsigned __int8 (__fastcall **)(__int64, char *, __int64))(*(_QWORD *)v39 + 24LL))(v39, "gvn", 3) )
        {
          goto LABEL_56;
        }
      }
      v69 = *(__int64 **)(a2 + 96);
      v67 = *(_QWORD *)(a2 + 24);
      sub_B176B0((__int64)&v80, (__int64)"gvn", (__int64)"LoadClobbered", 13, a3);
      sub_B18290((__int64)&v80, "load of type ", 0xDu);
      sub_B16360((__int64)&v76, "Type", 4, *(_QWORD *)(a3 + 8));
      v19 = sub_2445430((__int64)&v80, (__int64)&v76);
      sub_B18290(v19, " not eliminated", 0xFu);
      sub_B17B50(v19);
      if ( (_QWORD *)v78.m128i_i64[0] != v79 )
        j_j___libc_free_0(v78.m128i_u64[0]);
      if ( (__m128i *)v76.m128i_i64[0] != &v77 )
        j_j___libc_free_0(v76.m128i_u64[0]);
      v20 = *(_QWORD *)(*(_QWORD *)(a3 - 32) + 16LL);
      if ( v20 )
      {
        v66 = 0;
        v65 = v7;
        do
        {
          v21 = *(unsigned __int8 **)(v20 + 24);
          if ( (unsigned __int8 *)a3 != v21 && (unsigned __int8)(*v21 - 61) <= 1u )
          {
            v22 = sub_B43CB0(*(_QWORD *)(v20 + 24));
            if ( v22 == sub_B43CB0(a3) )
            {
              if ( (unsigned __int8)sub_B19DB0(v67, (__int64)v21, a3) )
              {
                if ( v66 )
                {
                  if ( !(unsigned __int8)sub_B19DB0(v67, (__int64)v66, (__int64)v21) )
                    v21 = v66;
                  v66 = v21;
                }
                else
                {
                  v66 = v21;
                }
              }
            }
          }
          v20 = *(_QWORD *)(v20 + 8);
        }
        while ( v20 );
        if ( v66 )
        {
LABEL_39:
          sub_B18290((__int64)&v80, " in favor of ", 0xDu);
          sub_B16080((__int64)&v76, "OtherAccess", 11, v66);
          sub_2445430((__int64)&v80, (__int64)&v76);
          if ( (_QWORD *)v78.m128i_i64[0] != v79 )
            j_j___libc_free_0(v78.m128i_u64[0]);
          if ( (__m128i *)v76.m128i_i64[0] != &v77 )
            j_j___libc_free_0(v76.m128i_u64[0]);
          goto LABEL_43;
        }
        v42 = *(_QWORD *)(a3 - 32);
        if ( *(_QWORD *)(v42 + 16) )
        {
          v43 = a3;
          v44 = *(_QWORD *)(v42 + 16);
          do
          {
            v45 = *(unsigned __int8 **)(v44 + 24);
            if ( (unsigned __int8 *)v43 != v45 && (unsigned __int8)(*v45 - 61) <= 1u )
            {
              v46 = sub_B43CB0(*(_QWORD *)(v44 + 24));
              if ( v46 == sub_B43CB0(v43) )
              {
                if ( (unsigned __int8)sub_D0EBA0((__int64)v45, v43, 0, v67, 0) )
                {
                  if ( v66 && !(unsigned __int8)sub_2789F00((__int64)v66, (__int64)v45, v43, v67) )
                  {
                    if ( !(unsigned __int8)sub_2789F00((__int64)v45, (__int64)v66, v43, v67) )
                    {
                      v7 = v65;
                      goto LABEL_43;
                    }
                  }
                  else
                  {
                    v66 = v45;
                  }
                }
              }
            }
            v44 = *(_QWORD *)(v44 + 8);
          }
          while ( v44 );
          v7 = v65;
          if ( !v66 )
            goto LABEL_43;
          goto LABEL_39;
        }
      }
LABEL_43:
      sub_B18290((__int64)&v80, " because it is clobbered by ", 0x1Cu);
      sub_B16080((__int64)&v76, "ClobberedBy", 11, (unsigned __int8 *)v74);
      sub_2445430((__int64)&v80, (__int64)&v76);
      if ( (_QWORD *)v78.m128i_i64[0] != v79 )
        j_j___libc_free_0(v78.m128i_u64[0]);
      if ( (__m128i *)v76.m128i_i64[0] != &v77 )
        j_j___libc_free_0(v76.m128i_u64[0]);
      sub_1049740(v69, (__int64)&v80);
      v23 = v83;
      v80.m128i_i64[0] = (__int64)&unk_49D9D40;
      v24 = &v83[10 * v84];
      if ( v83 != v24 )
      {
        do
        {
          v24 -= 10;
          v25 = v24[4];
          if ( (unsigned __int64 *)v25 != v24 + 6 )
            j_j___libc_free_0(v25);
          if ( (unsigned __int64 *)*v24 != v24 + 2 )
            j_j___libc_free_0(*v24);
        }
        while ( v23 != v24 );
        v24 = v83;
      }
      if ( v24 != (unsigned __int64 *)&v85 )
        _libc_free((unsigned __int64)v24);
      goto LABEL_56;
    }
    if ( a3 == v74 )
      goto LABEL_27;
    if ( !a5 )
      goto LABEL_27;
    v70 = sub_B46500((unsigned __int8 *)a3);
    if ( (unsigned __int8)v70 > (unsigned __int8)sub_B46500((unsigned __int8 *)v74) )
      goto LABEL_27;
    v71 = *(_QWORD *)(a3 + 8);
    v26 = sub_B43CB0(v74);
    v27 = sub_2A88720(v74, v71, v26);
    v28 = v71;
    if ( v27 )
    {
      v59 = *(_QWORD *)(a2 + 16);
      v60 = *(unsigned int *)(v59 + 1008);
      v61 = *(_QWORD *)(v59 + 992);
      if ( (_DWORD)v60 )
      {
        v62 = (v60 - 1) & (((unsigned int)v74 >> 9) ^ ((unsigned int)v74 >> 4));
        v63 = (__int64 *)(v61 + 16LL * v62);
        v64 = *v63;
        if ( v74 == *v63 )
        {
LABEL_114:
          if ( v63 != (__int64 *)(v61 + 16 * v60) )
          {
            v30 = *((_DWORD *)v63 + 2);
            if ( v30 >= 0 )
              goto LABEL_64;
          }
        }
        else
        {
          while ( v64 != -4096 )
          {
            v62 = (v60 - 1) & (v62 + v5);
            v63 = (__int64 *)(v61 + 16LL * v62);
            v64 = *v63;
            if ( v74 == *v63 )
              goto LABEL_114;
            ++v5;
          }
        }
      }
    }
    v29 = sub_2A89060(v71, a5, v74, v73);
    v28 = v71;
    v30 = v29;
LABEL_64:
    v72 = v30;
    v31.m128i_i64[0] = sub_9208B0(v73, v28);
    v80 = v31;
    v76.m128i_i64[0] = (unsigned __int64)(v31.m128i_i64[0] + 7) >> 3;
    v76.m128i_i8[8] = v31.m128i_i8[8];
    v32 = sub_CA1930(&v76);
    v33.m128i_i64[0] = (unsigned __int64)(sub_9208B0(v73, *(_QWORD *)(v74 + 8)) + 7) >> 3;
    v80 = v33;
    v34 = sub_CA1930(&v80);
    if ( v72 != -1 && ((_BYTE)qword_4FFB508 != 1 || v72 + v32 <= v34) )
    {
      *(_DWORD *)(v7 + 12) = v72;
      *(_DWORD *)(v7 + 8) = 1;
      *(_QWORD *)v7 = v74;
      *(_QWORD *)(v7 + 16) = 0;
      *(_QWORD *)(v7 + 24) = 0;
      *(_BYTE *)(v7 + 32) = 1;
      return v7;
    }
    v17 = *(_BYTE *)v74;
    goto LABEL_73;
  }
  if ( (a4 & 6) == 0 )
  {
    if ( (a4 & 7) == 0 )
      goto LABEL_4;
LABEL_13:
    BUG();
  }
  if ( v5 != 2 )
  {
    if ( v5 == 3 )
      sub_B43CC0(a3);
    goto LABEL_13;
  }
LABEL_4:
  v9 = (unsigned __int8 *)(a4 & 0xFFFFFFFFFFFFFFF8LL);
  sub_B43CC0(a3);
  if ( *v9 == 60
    || *v9 == 85
    && (v14 = *((_QWORD *)v9 - 4)) != 0
    && !*(_BYTE *)v14
    && *(_QWORD *)(v14 + 24) == *((_QWORD *)v9 + 10)
    && (*(_BYTE *)(v14 + 33) & 0x20) != 0
    && *(_DWORD *)(v14 + 36) == 211 )
  {
    v10 = sub_ACA8A0(*(__int64 ***)(a3 + 8));
    goto LABEL_22;
  }
  v10 = sub_D5D1D0(v9, *(__int64 **)(a2 + 32), *(__int64 ***)(a3 + 8));
  if ( !v10 )
  {
    v11 = *v9;
    if ( *v9 == 62 )
    {
      v12 = sub_B43CB0((__int64)v9);
      if ( (unsigned __int8)sub_2A88720(*((_QWORD *)v9 - 8), *(_QWORD *)(a3 + 8), v12) )
      {
        v13 = sub_B46500(v9);
        if ( (unsigned __int8)v13 >= (unsigned __int8)sub_B46500((unsigned __int8 *)a3) )
        {
          v10 = *((_QWORD *)v9 - 8);
          goto LABEL_22;
        }
      }
    }
    else if ( (_BYTE)v11 == 61 )
    {
      v35 = sub_B43CB0((__int64)v9);
      if ( (unsigned __int8)sub_2A88720(v9, *(_QWORD *)(a3 + 8), v35) )
      {
        v36 = sub_B46500(v9);
        if ( (unsigned __int8)v36 >= (unsigned __int8)sub_B46500((unsigned __int8 *)a3) )
        {
          *(_QWORD *)a1 = v9;
          *(_QWORD *)(a1 + 8) = 1;
          *(_QWORD *)(a1 + 16) = 0;
          *(_QWORD *)(a1 + 24) = 0;
          *(_BYTE *)(a1 + 32) = 1;
          return v7;
        }
      }
    }
    else if ( (_BYTE)v11 == 86 )
    {
      sub_D665A0(&v76, a3);
      v47 = *((_QWORD *)v9 - 8);
      v48 = *(_QWORD *)(a3 + 8);
      v49 = _mm_loadu_si128(&v77);
      v50 = _mm_loadu_si128(&v78);
      v51 = *(_QWORD **)(a2 + 320);
      v80.m128i_i64[1] = _mm_loadu_si128(&v76).m128i_i64[1];
      v81 = v49;
      v80.m128i_i64[0] = v47;
      v82 = v50;
      v52 = sub_2789F90(&v80, v48, (unsigned __int64)v9, v51);
      if ( v52 )
      {
        v75 = v52;
        v53 = *((_QWORD *)v9 - 4);
        v54 = _mm_loadu_si128(&v77);
        v55 = _mm_loadu_si128(&v78);
        v56 = *(_QWORD **)(a2 + 320);
        v80 = _mm_loadu_si128(&v76);
        v57 = *(_QWORD *)(a3 + 8);
        v80.m128i_i64[0] = v53;
        v81 = v54;
        v82 = v55;
        v58 = sub_2789F90(&v80, v57, (unsigned __int64)v9, v56);
        if ( v58 )
        {
          *(_QWORD *)a1 = v9;
          *(_QWORD *)(a1 + 8) = 4;
          *(_QWORD *)(a1 + 16) = v75;
          *(_QWORD *)(a1 + 24) = v58;
          *(_BYTE *)(a1 + 32) = 1;
          return v7;
        }
      }
    }
LABEL_56:
    *(_BYTE *)(v7 + 32) = 0;
    return v7;
  }
LABEL_22:
  *(_QWORD *)a1 = v10;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 32) = 1;
  return v7;
}
