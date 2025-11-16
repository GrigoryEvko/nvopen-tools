// Function: sub_39C9FF0
// Address: 0x39c9ff0
//
__int64 __fastcall sub_39C9FF0(__int64 *a1, __int64 a2, char a3)
{
  int v6; // r14d
  __int64 v7; // rax
  __int16 v8; // r14
  __int64 v9; // r15
  __int64 v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v14; // rbx
  __int64 v15; // rdx
  char v16; // al
  __int64 v17; // rax
  __int64 v18; // rax
  __m128i *v19; // rbx
  __int64 v20; // rdx
  _BYTE *v21; // rax
  _BYTE *v22; // r8
  size_t v23; // r9
  unsigned __int64 v24; // r15
  void *v25; // rcx
  _QWORD *v26; // rdi
  __int64 (*v27)(); // rax
  __int64 v28; // rsi
  unsigned int v29; // ecx
  __int64 v30; // rdi
  __int64 (*v31)(); // rax
  __int64 v32; // rax
  __int64 (*v33)(); // rax
  _QWORD *v34; // r15
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rdx
  unsigned int v38; // eax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 *v41; // r14
  void *v42; // rdi
  size_t v43; // rdx
  size_t v44; // rsi
  int v45; // ebx
  __int64 v46; // rax
  bool v47; // al
  __int64 v48; // rax
  __int64 *v49; // r13
  unsigned __int64 v50; // rdx
  const char *v51; // r8
  size_t v52; // r9
  __m128i *v53; // rax
  _BYTE *v54; // rdi
  void **v55; // rax
  __int64 v56; // r8
  __int64 v57; // rdi
  __int64 v58; // rax
  __int64 v59; // r14
  __int64 v60; // rax
  __int64 v61; // rax
  size_t v62; // rdx
  void *v63; // rax
  _QWORD *v64; // rdi
  __int64 v65; // [rsp+8h] [rbp-158h]
  unsigned int v66; // [rsp+10h] [rbp-150h]
  char v67; // [rsp+17h] [rbp-149h]
  size_t v68; // [rsp+18h] [rbp-148h]
  size_t v69; // [rsp+20h] [rbp-140h]
  size_t v70; // [rsp+20h] [rbp-140h]
  int v71; // [rsp+28h] [rbp-138h]
  _BYTE *v72; // [rsp+28h] [rbp-138h]
  unsigned int v73; // [rsp+28h] [rbp-138h]
  const char *v74; // [rsp+28h] [rbp-138h]
  __int64 *v75; // [rsp+30h] [rbp-130h]
  __int64 *v76; // [rsp+38h] [rbp-128h]
  __m128i *v77; // [rsp+38h] [rbp-128h]
  unsigned int v78; // [rsp+48h] [rbp-118h] BYREF
  unsigned int v79; // [rsp+4Ch] [rbp-114h] BYREF
  void *dest; // [rsp+50h] [rbp-110h] BYREF
  size_t v81; // [rsp+58h] [rbp-108h]
  _QWORD v82[2]; // [rsp+60h] [rbp-100h] BYREF
  _QWORD *v83; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v84; // [rsp+78h] [rbp-E8h]
  _QWORD v85[8]; // [rsp+80h] [rbp-E0h] BYREF
  void **p_src; // [rsp+C0h] [rbp-A0h] BYREF
  size_t n; // [rsp+C8h] [rbp-98h]
  __m128i src; // [rsp+D0h] [rbp-90h] BYREF
  int v89; // [rsp+10Ch] [rbp-54h]
  __int64 **v90; // [rsp+128h] [rbp-38h]

  v6 = -(*(_WORD *)(*(_QWORD *)a2 + 32LL) == 0);
  v76 = a1 + 11;
  v7 = sub_145CDC0(0x30u, a1 + 11);
  v8 = (v6 & 0x2F) + 5;
  v9 = v7;
  if ( v7 )
  {
    *(_QWORD *)(v7 + 8) = 0;
    *(_QWORD *)v7 = v7 | 4;
    *(_QWORD *)(v7 + 16) = 0;
    *(_DWORD *)(v7 + 24) = -1;
    *(_WORD *)(v7 + 28) = v8;
    *(_BYTE *)(v7 + 30) = 0;
    *(_QWORD *)(v7 + 32) = 0;
    *(_QWORD *)(v7 + 40) = 0;
  }
  sub_39A55B0((__int64)a1, *(unsigned __int8 **)a2, (unsigned __int8 *)v7);
  if ( a3 )
  {
    sub_39C8370(a1, a2, v9);
    return v9;
  }
  if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1[24] + 232) + 504LL) - 34) > 1 )
  {
    LODWORD(v11) = *(_DWORD *)(a2 + 24);
    if ( (_DWORD)v11 != -1 )
      goto LABEL_17;
  }
  else
  {
    v10 = a1[25];
    if ( *(_DWORD *)(v10 + 6584) == 1 && *(_WORD *)(*(_QWORD *)a2 + 32LL) )
    {
      if ( sub_39C9F10(v10 + 6592, a2) )
      {
        v41 = *(__int64 **)(a1[24] + 264);
        v42 = *(void **)(*(_QWORD *)a2 + 8 * (1LL - *(unsigned int *)(*(_QWORD *)a2 + 8LL)));
        if ( v42 )
        {
          v42 = (void *)sub_161E970((__int64)v42);
          v44 = v43;
        }
        else
        {
          v44 = 0;
        }
        v45 = sub_39C81B0(v42, v44, v41);
        v46 = sub_15AB1E0(*(_BYTE **)(*(_QWORD *)a2 - 8LL * *(unsigned int *)(*(_QWORD *)a2 + 8LL)));
        v47 = sub_15B1050(v46, **(_QWORD **)(a1[24] + 264));
        if ( v45 >= 0 && v47 )
        {
          v48 = sub_145CDC0(0x10u, v76);
          v49 = (__int64 *)v48;
          if ( v48 )
          {
            *(_QWORD *)v48 = 0;
            *(_DWORD *)(v48 + 8) = 0;
          }
          BYTE2(p_src) = 0;
          sub_39A3560((__int64)a1, (__int64 *)v48, 0, (__int64)&p_src, 3);
          v51 = sub_1649960(**(_QWORD **)(a1[24] + 264));
          v52 = v50;
          if ( !v51 )
          {
            LOBYTE(v82[0]) = 0;
            dest = v82;
            v81 = 0;
LABEL_70:
            v83 = v85;
            sub_39C7500((__int64 *)&v83, "_param_", (__int64)"");
            v53 = (__m128i *)sub_2241130((unsigned __int64 *)&v83, 0, 0, dest, v81);
            p_src = (void **)&src;
            if ( (__m128i *)v53->m128i_i64[0] == &v53[1] )
            {
              src = _mm_loadu_si128(v53 + 1);
            }
            else
            {
              p_src = (void **)v53->m128i_i64[0];
              src.m128i_i64[0] = v53[1].m128i_i64[0];
            }
            n = v53->m128i_u64[1];
            v53->m128i_i64[0] = (__int64)v53[1].m128i_i64;
            v53->m128i_i64[1] = 0;
            v53[1].m128i_i8[0] = 0;
            v54 = dest;
            v55 = (void **)dest;
            if ( p_src == (void **)&src )
            {
              v62 = n;
              if ( n )
              {
                if ( n == 1 )
                  *(_BYTE *)dest = src.m128i_i8[0];
                else
                  memcpy(dest, &src, n);
                v54 = dest;
                v62 = n;
              }
              v81 = v62;
              v54[v62] = 0;
              v55 = p_src;
              goto LABEL_76;
            }
            if ( dest == v82 )
            {
              dest = p_src;
              v81 = n;
              v82[0] = src.m128i_i64[0];
            }
            else
            {
              v56 = v82[0];
              dest = p_src;
              v81 = n;
              v82[0] = src.m128i_i64[0];
              if ( v55 )
              {
                p_src = v55;
                src.m128i_i64[0] = v56;
LABEL_76:
                n = 0;
                *(_BYTE *)v55 = 0;
                if ( p_src != (void **)&src )
                  j_j___libc_free_0((unsigned __int64)p_src);
                if ( v83 != v85 )
                  j_j___libc_free_0((unsigned __int64)v83);
                v57 = *(_QWORD *)(a1[24] + 248);
                LODWORD(v83) = v45;
                src.m128i_i16[0] = 2564;
                p_src = &dest;
                n = (size_t)v83;
                v58 = sub_38BF510(v57, (__int64)&p_src);
                sub_39A3990((__int64)a1, v49, 0, 1, v58);
                sub_39A4C90(a1, v9, 2, (__int64 **)v49);
                BYTE2(p_src) = 0;
                sub_39A3560((__int64)a1, (__int64 *)(v9 + 8), 51, (__int64)&p_src, 7);
                if ( dest != v82 )
                  j_j___libc_free_0((unsigned __int64)dest);
                return v9;
              }
            }
            p_src = (void **)&src;
            v55 = (void **)&src;
            goto LABEL_76;
          }
          p_src = (void **)v50;
          dest = v82;
          if ( v50 > 0xF )
          {
            v70 = v50;
            v74 = v51;
            v63 = (void *)sub_22409D0((__int64)&dest, (unsigned __int64 *)&p_src, 0);
            v51 = v74;
            v52 = v70;
            dest = v63;
            v64 = v63;
            v82[0] = p_src;
          }
          else
          {
            if ( v50 == 1 )
            {
              LOBYTE(v82[0]) = *v51;
LABEL_69:
              v81 = (size_t)p_src;
              *((_BYTE *)p_src + (_QWORD)dest) = 0;
              goto LABEL_70;
            }
            if ( !v50 )
              goto LABEL_69;
            v64 = v82;
          }
          memcpy(v64, v51, v52);
          goto LABEL_69;
        }
      }
      v11 = *(unsigned int *)(a2 + 24);
      if ( (_DWORD)v11 != -1 )
      {
        if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1[24] + 232) + 504LL) - 34) <= 1 )
        {
          v10 = a1[25];
          goto LABEL_7;
        }
LABEL_17:
        sub_39C9720((__int64)a1, v9, 2, v11);
        return v9;
      }
    }
    else
    {
      v11 = *(unsigned int *)(a2 + 24);
      if ( (_DWORD)v11 != -1 )
      {
LABEL_7:
        v12 = *(unsigned int *)(*(_QWORD *)(v10 + 1192) + 32 * v11 + 24);
        LODWORD(p_src) = 65542;
        sub_39A3560((__int64)a1, (__int64 *)(v9 + 8), 2, (__int64)&p_src, v12);
        return v9;
      }
    }
  }
  v14 = *(_QWORD *)(a2 + 32);
  if ( v14 )
  {
    v15 = *(_QWORD *)(v14 + 32);
    v16 = *(_BYTE *)v15;
    if ( !*(_BYTE *)v15 )
    {
      sub_39C9A70(a1, a2, v9, ((unsigned __int64)*(unsigned int *)(v15 + 8) << 32) | (*(_BYTE *)(v15 + 40) != 1));
      return v9;
    }
    if ( v16 != 1 )
    {
      if ( v16 == 3 )
      {
        sub_39A4D60(a1, v9, v15);
      }
      else if ( v16 == 2 )
      {
        v17 = sub_3988770(a2);
        sub_39A5150((__int64)a1, v9, *(_QWORD *)(*(_QWORD *)(v14 + 32) + 24LL), v17);
      }
      return v9;
    }
    if ( !*(_DWORD *)(a2 + 48)
      || (v59 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL)) == 0
      || !(unsigned int)((__int64)(*(_QWORD *)(v59 + 32) - *(_QWORD *)(v59 + 24)) >> 3) )
    {
      v60 = sub_3988770(a2);
      sub_39A3830((__int64)a1, v9, *(_QWORD *)(v14 + 32), v60);
      return v9;
    }
    v61 = sub_145CDC0(0x10u, v76);
    if ( v61 )
    {
      *(_QWORD *)v61 = 0;
      *(_DWORD *)(v61 + 8) = 0;
    }
    sub_39A1E10((__int64)&p_src, a1[24], (__int64)a1, v61);
    sub_399FD50((__int64)&p_src, v59);
    sub_399F670(&p_src, *(_QWORD *)(*(_QWORD *)(v14 + 32) + 24LL));
    v83 = *(_QWORD **)(v59 + 24);
    v84 = *(_QWORD *)(v59 + 32);
    sub_399FAC0(&p_src, &v83, 0);
    goto LABEL_54;
  }
  if ( !*(_DWORD *)(a2 + 48) )
    return v9;
  v18 = sub_145CDC0(0x10u, v76);
  v75 = (__int64 *)v18;
  if ( v18 )
  {
    *(_QWORD *)v18 = 0;
    *(_DWORD *)(v18 + 8) = 0;
  }
  sub_39A1E10((__int64)&p_src, a1[24], (__int64)a1, v18);
  v67 = 0;
  v19 = sub_39888D0(a2);
  v20 *= 16;
  v77 = (__m128i *)((char *)v19 + v20);
  if ( &v19->m128i_i8[v20] != (__int8 *)v19 )
  {
    v65 = v9;
    while ( 1 )
    {
      v32 = a1[24];
      v78 = 0;
      v33 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)(v32 + 264) + 16LL) + 48LL);
      if ( v33 == sub_1D90020 )
        BUG();
      v34 = (_QWORD *)v19->m128i_i64[1];
      v35 = v33();
      v71 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, unsigned int *))(*(_QWORD *)v35 + 176LL))(
              v35,
              *(_QWORD *)(a1[24] + 264),
              v19->m128i_u32[0],
              &v78);
      sub_399FD50((__int64)&p_src, (__int64)v34);
      v37 = 2;
      v83 = v85;
      v85[0] = 35;
      v85[1] = v71;
      v84 = 0x800000002LL;
      if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1[24] + 232) + 504LL) - 34) <= 1 && *(_DWORD *)(a1[25] + 6584) == 1 )
      {
        v39 = sub_15C4660(v34, &v79);
        v37 = (unsigned int)v84;
        if ( v34 != (_QWORD *)v39 )
        {
          v34 = (_QWORD *)v39;
          v37 = (unsigned int)v84;
          v66 = v79;
          if ( !v67 )
            v67 = 1;
        }
      }
      if ( v34 )
      {
        v21 = (_BYTE *)v34[4];
        v22 = (_BYTE *)v34[3];
        v23 = v21 - v22;
        v24 = (v21 - v22) >> 3;
        if ( v24 > (unsigned __int64)HIDWORD(v84) - v37 )
        {
          v68 = v21 - v22;
          v69 = (size_t)v21;
          v72 = v22;
          sub_16CD150((__int64)&v83, v85, v24 + v37, 8, (int)v22, v23);
          v37 = (unsigned int)v84;
          v23 = v68;
          v21 = (_BYTE *)v69;
          v22 = v72;
        }
        v25 = v83;
        if ( v22 != v21 )
        {
          memcpy(&v83[v37], v22, v23);
          LODWORD(v37) = v84;
          v25 = v83;
        }
        LODWORD(v84) = v24 + v37;
        v37 = (unsigned int)(v24 + v37);
      }
      else
      {
        v25 = v83;
      }
      v26 = (_QWORD *)a1[24];
      dest = v25;
      v81 = (size_t)v25 + 8 * v37;
      v89 = 2;
      v27 = *(__int64 (**)())(*v26 + 184LL);
      if ( v27 == sub_215B780 )
        goto LABEL_30;
      v36 = v27();
      if ( !v36 )
        break;
      sub_39A39D0((__int64)a1, v75, v36);
LABEL_33:
      sub_399FAC0(&p_src, &dest, 0);
      if ( v83 != v85 )
        _libc_free((unsigned __int64)v83);
      if ( v77 == ++v19 )
      {
        v9 = v65;
        goto LABEL_49;
      }
    }
    v26 = (_QWORD *)a1[24];
LABEL_30:
    v28 = 0;
    v29 = v78;
    v30 = *(_QWORD *)(v26[33] + 16LL);
    v31 = *(__int64 (**)())(*(_QWORD *)v30 + 112LL);
    if ( v31 != sub_1D00B10 )
    {
      v73 = v78;
      v40 = ((__int64 (__fastcall *)(__int64, _QWORD))v31)(v30, 0);
      v29 = v73;
      v28 = v40;
    }
    sub_399F750((__int64)&p_src, v28, (unsigned __int64 **)&dest, v29);
    goto LABEL_33;
  }
LABEL_49:
  if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1[24] + 232) + 504LL) - 34) <= 1 && *(_DWORD *)(a1[25] + 6584) == 1 )
  {
    v38 = 6;
    if ( v67 )
      v38 = v66;
    LODWORD(v83) = 65547;
    sub_39A3560((__int64)a1, (__int64 *)(v9 + 8), 51, (__int64)&v83, v38);
  }
LABEL_54:
  sub_399FD30((__int64)&p_src);
  sub_39A4520(a1, v9, 2, v90);
  if ( (unsigned __int64 *)n != &src.m128i_u64[1] )
    _libc_free(n);
  return v9;
}
