// Function: sub_26D56A0
// Address: 0x26d56a0
//
__int64 __fastcall sub_26D56A0(__int64 a1, __int64 a2, __int64 a3, __m128i a4)
{
  __int64 v5; // r12
  __int64 v6; // r15
  __int64 v7; // rbx
  char v8; // al
  __int64 v9; // rax
  __int64 v10; // rcx
  __m128i *v11; // rsi
  __m128i *v12; // rsi
  __int64 v13; // r9
  int v14; // eax
  int v15; // ecx
  __int64 v16; // r12
  __int64 v17; // r13
  __int64 v18; // rcx
  _QWORD *v19; // r8
  __int64 v20; // r9
  const __m128i *v21; // rdi
  __m128i *v22; // rax
  signed __int64 v23; // rdx
  bool v24; // zf
  unsigned __int8 v25; // bl
  __m128i *v26; // rdi
  __int64 v27; // rdi
  __int64 v28; // rsi
  float v30; // xmm0_4
  unsigned __int64 v31; // rdi
  _QWORD **v32; // rbx
  _QWORD *v33; // r13
  __int64 v34; // rax
  float v35; // xmm0_4
  unsigned __int64 v36; // r12
  __m128i *v37; // rdi
  _BYTE **v38; // r13
  _BYTE **v39; // rbx
  __int64 v40; // rcx
  __m128i *v41; // rsi
  __m128i *v42; // rsi
  __int64 v43; // rax
  __int64 v44; // r9
  _BYTE *v45; // rdx
  __int64 v46; // rax
  unsigned __int64 v47; // rbx
  _QWORD *v48; // rax
  unsigned __int64 v49; // rax
  unsigned int v50; // eax
  _BYTE **v51; // r12
  _BYTE **v52; // r13
  __int64 v53; // rcx
  __m128i *v54; // rsi
  __m128i *v55; // rsi
  __int64 v56; // rax
  __int64 v57; // r9
  _BYTE *v58; // rdx
  __int64 v59; // rax
  unsigned __int8 v61; // [rsp+20h] [rbp-1A0h]
  _QWORD **v62; // [rsp+20h] [rbp-1A0h]
  unsigned __int8 v63; // [rsp+2Fh] [rbp-191h]
  __int64 v64; // [rsp+38h] [rbp-188h]
  _QWORD **v65; // [rsp+40h] [rbp-180h]
  unsigned int v66; // [rsp+48h] [rbp-178h]
  unsigned int v67; // [rsp+4Ch] [rbp-174h]
  __int64 v68; // [rsp+50h] [rbp-170h]
  __int64 v69; // [rsp+50h] [rbp-170h]
  unsigned __int64 v71; // [rsp+68h] [rbp-158h] BYREF
  _QWORD **v72; // [rsp+70h] [rbp-150h] BYREF
  _QWORD **v73; // [rsp+78h] [rbp-148h]
  const __m128i *v74; // [rsp+90h] [rbp-130h] BYREF
  __m128i *v75; // [rsp+98h] [rbp-128h]
  __m128i *v76; // [rsp+A0h] [rbp-120h]
  __m128i v77; // [rsp+B0h] [rbp-110h] BYREF
  __m128i v78; // [rsp+C0h] [rbp-100h] BYREF
  __m128i v79; // [rsp+D0h] [rbp-F0h] BYREF
  __m128i v80; // [rsp+E0h] [rbp-E0h]
  __int64 v81; // [rsp+F0h] [rbp-D0h] BYREF
  unsigned __int64 v82; // [rsp+F8h] [rbp-C8h] BYREF
  __int64 v83; // [rsp+100h] [rbp-C0h]
  __int64 v84; // [rsp+108h] [rbp-B8h]
  __int64 v85; // [rsp+110h] [rbp-B0h] BYREF
  __int64 v86; // [rsp+118h] [rbp-A8h]
  __int64 v87; // [rsp+120h] [rbp-A0h]
  unsigned int v88; // [rsp+128h] [rbp-98h]
  __int128 *v89; // [rsp+130h] [rbp-90h]
  __int64 v90; // [rsp+138h] [rbp-88h]
  __int128 v91; // [rsp+140h] [rbp-80h] BYREF
  __m128i v92; // [rsp+150h] [rbp-70h] BYREF
  char *v93; // [rsp+160h] [rbp-60h]
  char v94; // [rsp+170h] [rbp-50h] BYREF

  v5 = *(_QWORD *)(a2 + 80);
  v74 = 0;
  v75 = 0;
  v76 = 0;
  v68 = a2 + 72;
  if ( a2 + 72 != v5 )
  {
    while ( 1 )
    {
      if ( !v5 )
        BUG();
      v6 = *(_QWORD *)(v5 + 32);
      v7 = v5 + 24;
      if ( v6 != v5 + 24 )
        break;
LABEL_19:
      v5 = *(_QWORD *)(v5 + 8);
      if ( v5 == v68 )
        goto LABEL_20;
    }
    while ( 1 )
    {
      if ( !v6 )
        BUG();
      v8 = *(_BYTE *)(v6 - 24);
      if ( v8 == 85 )
      {
        v9 = *(_QWORD *)(v6 - 56);
        if ( v9 && !*(_BYTE *)v9 && *(_QWORD *)(v9 + 24) == *(_QWORD *)(v6 + 56) && (*(_BYTE *)(v9 + 33) & 0x20) != 0 )
          goto LABEL_7;
      }
      else if ( v8 != 40 && v8 != 34 )
      {
        goto LABEL_7;
      }
      if ( sub_26CAFB0((_QWORD *)a1, (__int64)&v77, v6 - 24) )
      {
        v11 = v75;
        if ( v75 == v76 )
        {
          sub_26BA1A0((__int64)&v74, v75, &v77);
          v12 = v75;
        }
        else
        {
          if ( v75 )
          {
            *v75 = _mm_loadu_si128(&v77);
            v11[1] = _mm_loadu_si128(&v78);
            v11 = v75;
          }
          v12 = v11 + 2;
          v75 = v12;
        }
        v91 = (__int128)v12[-2];
        v13 = v12[-1].m128i_i64[1];
        v92.m128i_i64[0] = v12[-1].m128i_i64[0];
        v92.m128i_i64[1] = v13;
        sub_26BCD60((__int64)v74, (((char *)v12 - (char *)v74) >> 5) - 1, 0, v10, v92.m128i_i64[0], v13, a4, v91);
        v6 = *(_QWORD *)(v6 + 8);
        if ( v6 == v7 )
          goto LABEL_19;
      }
      else
      {
LABEL_7:
        v6 = *(_QWORD *)(v6 + 8);
        if ( v6 == v7 )
          goto LABEL_19;
      }
    }
  }
LABEL_20:
  v14 = sub_B2BED0(a2);
  v66 = -1;
  v15 = dword_4FF75C8[32];
  if ( !*(_QWORD *)(a1 + 1712) )
  {
    v50 = LODWORD(qword_4FF76A0[17]) * v14;
    if ( v50 > dword_4FF74E8[32] )
      v50 = dword_4FF74E8[32];
    if ( v50 >= dword_4FF75C8[32] )
      v15 = v50;
    v66 = v15;
  }
  v85 = 0;
  v89 = &v91;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v90 = 0;
  v61 = 0;
  if ( v75 != v74 )
  {
LABEL_30:
    while ( 2 )
    {
      if ( (unsigned int)sub_B2BED0(a2) >= v66 )
        goto LABEL_44;
      v21 = v74;
      v22 = v75;
      v79 = _mm_loadu_si128(v74);
      v80 = _mm_loadu_si128(v74 + 1);
      if ( (char *)v75 - (char *)v74 > 32 )
      {
        v91 = (__int128)_mm_loadu_si128(v75 - 2);
        v23 = ((char *)&v75[-2] - (char *)v74) >> 5;
        v92 = _mm_loadu_si128(v75 - 1);
        v75[-2] = _mm_loadu_si128(v74);
        v22[-1] = _mm_loadu_si128(v21 + 1);
        sub_26BD170((__int64)v21, 0, v23, a4, v18, v19, v20, v91);
        v22 = v75;
      }
      v16 = v79.m128i_i64[0];
      v75 = v22 - 2;
      v17 = *(_QWORD *)(v79.m128i_i64[0] - 32);
      if ( v17 && !*(_BYTE *)v17 && *(_QWORD *)(v17 + 24) == *(_QWORD *)(v79.m128i_i64[0] + 80) )
      {
        if ( a2 == v17 )
          goto LABEL_29;
        if ( !sub_B491E0(v79.m128i_i64[0]) )
        {
          if ( !sub_B92180(v17) || sub_B2FC80(v17) )
          {
LABEL_27:
            if ( *(_DWORD *)(a1 + 1520) == 1 )
            {
              v47 = sub_D844E0(*(_QWORD *)(a1 + 1280));
              v48 = sub_26CAE90((_QWORD *)a1, v16);
              sub_26CF830(a1, v16, (unsigned __int64)v48, a3, v47);
            }
            goto LABEL_29;
          }
          v24 = *(_BYTE *)(a1 + 1705) == 0;
          *(_QWORD *)&v91 = &v92;
          *((_QWORD *)&v91 + 1) = 0x800000000LL;
          if ( v24 )
          {
            v25 = sub_26C3F00(a1, (__int64)&v79, (__int64)&v91);
            if ( !v25 )
            {
              if ( *(_QWORD *)(a1 + 1512) )
                goto LABEL_41;
              goto LABEL_94;
            }
            v26 = (__m128i *)v91;
            v51 = (_BYTE **)(v91 + 8LL * DWORD2(v91));
            if ( v51 == (_BYTE **)v91 )
            {
              v61 = v25;
            }
            else
            {
              v52 = (_BYTE **)v91;
              do
              {
                v58 = *v52;
                if ( (**v52 != 85
                   || (v59 = *((_QWORD *)v58 - 4)) == 0
                   || *(_BYTE *)v59
                   || *(_QWORD *)(v59 + 24) != *((_QWORD *)v58 + 10)
                   || (*(_BYTE *)(v59 + 33) & 0x20) == 0)
                  && sub_26CAFB0((_QWORD *)a1, (__int64)&v77, (__int64)v58) )
                {
                  v54 = v75;
                  if ( v75 == v76 )
                  {
                    sub_26BA1A0((__int64)&v74, v75, &v77);
                    v55 = v75;
                  }
                  else
                  {
                    if ( v75 )
                    {
                      *v75 = _mm_loadu_si128(&v77);
                      v54[1] = _mm_loadu_si128(&v78);
                      v54 = v75;
                    }
                    v55 = v54 + 2;
                    v75 = v55;
                  }
                  v56 = v55[-2].m128i_i64[0];
                  v82 = v55[-2].m128i_u64[1];
                  v81 = v56;
                  v57 = v55[-1].m128i_i64[1];
                  v83 = v55[-1].m128i_i64[0];
                  v84 = v57;
                  sub_26BCD60(
                    (__int64)v74,
                    (((char *)v55 - (char *)v74) >> 5) - 1,
                    0,
                    v53,
                    v83,
                    v57,
                    a4,
                    __PAIR128__(v82, v56));
                }
                ++v52;
              }
              while ( v51 != v52 );
              v26 = (__m128i *)v91;
              v61 = v25;
            }
          }
          else
          {
            if ( *(_QWORD *)(a1 + 1512) )
              goto LABEL_29;
LABEL_94:
            v81 = v16;
            v82 = v79.m128i_u64[1];
            sub_26D54A0((__int64)&v85, &v81, (__int64 *)&v82);
LABEL_41:
            v26 = (__m128i *)v91;
          }
          if ( v26 == &v92 )
            goto LABEL_29;
          _libc_free((unsigned __int64)v26);
          if ( v75 == v74 )
          {
LABEL_44:
            if ( unk_4F838D3 )
              goto LABEL_45;
            goto LABEL_107;
          }
          continue;
        }
      }
      else if ( !sub_B491E0(v79.m128i_i64[0]) )
      {
        goto LABEL_27;
      }
      break;
    }
    v71 = 0;
    sub_26CB210((__int64)&v72, a1, v16, (__int64 *)&v71);
    v64 = v71;
    a4 = 0;
    if ( (v71 & 0x8000000000000000LL) != 0LL )
      v30 = (float)(int)(v71 & 1 | (v71 >> 1)) + (float)(int)(v71 & 1 | (v71 >> 1));
    else
      v30 = (float)(int)v71;
    *(float *)a4.m128i_i32 = v30 * *(float *)&v80.m128i_i32[2];
    if ( *(float *)a4.m128i_i32 >= 9.223372e18 )
    {
      *(float *)a4.m128i_i32 = *(float *)a4.m128i_i32 - 9.223372e18;
      v71 = (unsigned int)(int)*(float *)a4.m128i_i32;
      v71 ^= 0x8000000000000000LL;
    }
    else
    {
      v71 = (unsigned int)(int)*(float *)a4.m128i_i32;
    }
    v31 = (unsigned __int64)v72;
    v67 = 0;
    v65 = v73;
    v32 = v72;
    if ( v73 == v72 )
      goto LABEL_90;
    v69 = v16;
    while ( 1 )
    {
      v33 = *v32;
      if ( *(_DWORD *)(a1 + 1520) == 1 )
        goto LABEL_98;
      while ( 1 )
      {
        v34 = sub_EF9210(v33);
        a4 = 0;
        if ( v34 < 0 )
          v35 = (float)(v34 & 1 | (unsigned int)((unsigned __int64)v34 >> 1))
              + (float)(v34 & 1 | (unsigned int)((unsigned __int64)v34 >> 1));
        else
          v35 = (float)(int)v34;
        *(float *)a4.m128i_i32 = v35 * *(float *)&v80.m128i_i32[2];
        if ( *(float *)a4.m128i_i32 >= 9.223372e18 )
        {
          *(float *)a4.m128i_i32 = *(float *)a4.m128i_i32 - 9.223372e18;
          v36 = (unsigned int)(int)*(float *)a4.m128i_i32 ^ 0x8000000000000000LL;
        }
        else
        {
          v36 = (unsigned int)(int)*(float *)a4.m128i_i32;
        }
        if ( (unsigned int)qword_4FF71E8 <= v67 && 100 * v36 < v64 * (unsigned __int64)(unsigned int)dword_4FF72C8
          || !sub_D84440(*(_QWORD *)(a1 + 1280), v36) )
        {
LABEL_89:
          v31 = (unsigned __int64)v72;
LABEL_90:
          if ( v31 )
            j_j___libc_free_0(v31);
LABEL_29:
          if ( v75 == v74 )
            goto LABEL_44;
          goto LABEL_30;
        }
        v24 = *(_BYTE *)(a1 + 1705) == 0;
        v79.m128i_i64[1] = (__int64)v33;
        *((_QWORD *)&v91 + 1) = 0x800000000LL;
        *(_QWORD *)&v91 = &v92;
        v79.m128i_i64[0] = v69;
        v80.m128i_i64[0] = v36;
        if ( !v24 || !(_DWORD)qword_4FF62C8 )
        {
          if ( *(_QWORD *)(a1 + 1512) )
            goto LABEL_96;
LABEL_102:
          v82 = (unsigned __int64)v33;
          v81 = v69;
          sub_26D54A0((__int64)&v85, &v81, (__int64 *)&v82);
LABEL_103:
          v37 = (__m128i *)v91;
          goto LABEL_87;
        }
        v63 = sub_26CC080(a1, a2, v79.m128i_i64, v64, &v71, (__int64)&v91);
        if ( !v63 )
        {
          if ( !*(_QWORD *)(a1 + 1512) )
            goto LABEL_102;
          goto LABEL_103;
        }
        v37 = (__m128i *)v91;
        v38 = (_BYTE **)(v91 + 8LL * DWORD2(v91));
        if ( v38 != (_BYTE **)v91 )
        {
          v62 = v32;
          v39 = (_BYTE **)v91;
          do
          {
            while ( 1 )
            {
              v45 = *v39;
              if ( **v39 != 85 )
                break;
              v46 = *((_QWORD *)v45 - 4);
              if ( !v46
                || *(_BYTE *)v46
                || *(_QWORD *)(v46 + 24) != *((_QWORD *)v45 + 10)
                || (*(_BYTE *)(v46 + 33) & 0x20) == 0 )
              {
                break;
              }
              if ( v38 == ++v39 )
                goto LABEL_85;
            }
            if ( sub_26CAFB0((_QWORD *)a1, (__int64)&v77, (__int64)v45) )
            {
              v41 = v75;
              if ( v75 == v76 )
              {
                sub_26BA1A0((__int64)&v74, v75, &v77);
                v42 = v75;
              }
              else
              {
                if ( v75 )
                {
                  *v75 = _mm_loadu_si128(&v77);
                  v41[1] = _mm_loadu_si128(&v78);
                  v41 = v75;
                }
                v42 = v41 + 2;
                v75 = v42;
              }
              v43 = v42[-2].m128i_i64[0];
              v82 = v42[-2].m128i_u64[1];
              v81 = v43;
              v44 = v42[-1].m128i_i64[1];
              v83 = v42[-1].m128i_i64[0];
              v84 = v44;
              sub_26BCD60(
                (__int64)v74,
                (((char *)v42 - (char *)v74) >> 5) - 1,
                0,
                v40,
                v83,
                v44,
                a4,
                __PAIR128__(v82, v43));
            }
            ++v39;
          }
          while ( v38 != v39 );
LABEL_85:
          v32 = v62;
          v37 = (__m128i *)v91;
        }
        ++v67;
        v61 = v63;
LABEL_87:
        if ( v37 != &v92 )
          break;
        while ( 1 )
        {
LABEL_96:
          if ( v65 == ++v32 )
            goto LABEL_89;
          v33 = *v32;
          if ( *(_DWORD *)(a1 + 1520) != 1 )
            break;
LABEL_98:
          v49 = sub_D844E0(*(_QWORD *)(a1 + 1280));
          sub_26CF830(a1, v69, (unsigned __int64)v33, a3, v49);
        }
      }
      _libc_free((unsigned __int64)v37);
      if ( v65 == ++v32 )
        goto LABEL_89;
    }
  }
  v61 = unk_4F838D3;
  if ( unk_4F838D3 )
  {
    v61 = 0;
    v27 = 0;
    v28 = 0;
  }
  else
  {
LABEL_107:
    sub_26C1C00((__int64)&v91, (__int64)&v85);
    sub_26C8F20(a1, (__int64)&v91, (unsigned __int8 *)a2);
    if ( v93 != &v94 )
      _libc_free((unsigned __int64)v93);
    sub_C7D6A0(*((__int64 *)&v91 + 1), 16LL * v92.m128i_u32[2], 8);
LABEL_45:
    if ( v89 != &v91 )
      _libc_free((unsigned __int64)v89);
    v27 = v86;
    v28 = 16LL * v88;
  }
  sub_C7D6A0(v27, v28, 8);
  if ( v74 )
    j_j___libc_free_0((unsigned __int64)v74);
  return v61;
}
