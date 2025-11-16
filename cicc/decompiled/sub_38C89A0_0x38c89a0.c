// Function: sub_38C89A0
// Address: 0x38c89a0
//
__int64 __fastcall sub_38C89A0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        const __m128i *a6,
        unsigned int a7)
{
  _QWORD *v10; // rbx
  size_t v11; // rdx
  unsigned __int64 v12; // r14
  __int64 v13; // rdx
  __int64 v14; // rdi
  size_t v15; // r14
  size_t v16; // r8
  size_t v17; // r9
  unsigned int v18; // r11d
  __int8 *v19; // rcx
  unsigned int v20; // r14d
  __int64 v21; // r12
  int v22; // r15d
  size_t v23; // rbx
  __int8 *v24; // r13
  int v25; // r15d
  __int8 *v26; // r9
  size_t v27; // r13
  __m128i *v28; // rax
  unsigned __int8 *v29; // rdi
  __m128i *v30; // rax
  __int64 v31; // rcx
  unsigned int v32; // eax
  const char *v33; // rcx
  __int64 v34; // rdx
  __int64 v35; // r13
  unsigned int v36; // r14d
  __int64 v37; // rax
  __int64 v38; // rbx
  unsigned int v40; // r9d
  __int64 *v41; // r11
  __int64 v42; // rax
  __m128i *v43; // rdi
  char v44; // dl
  __int64 v45; // rdx
  unsigned __int64 v46; // rsi
  __int64 v47; // rsi
  unsigned __int64 *v48; // r14
  unsigned __int64 *v49; // rbx
  __int64 v50; // rax
  size_t v51; // rdx
  unsigned int v52; // r9d
  __int64 *v53; // r11
  __int64 v54; // r8
  int v55; // eax
  __int64 v56; // rax
  __int64 v57; // r8
  __int64 v58; // rax
  __m128i v59; // xmm3
  unsigned __int8 *v60; // rax
  __m128i *v61; // rdi
  char *v62; // rax
  bool v63; // zf
  __m128i *v64; // rax
  __m128i *v65; // rax
  __m128i *v66; // rdi
  __m128i *v67; // rdi
  unsigned __int8 *v68; // rax
  __int64 v69; // [rsp+8h] [rbp-1E8h]
  __int64 *v70; // [rsp+10h] [rbp-1E0h]
  __int64 *v71; // [rsp+10h] [rbp-1E0h]
  __int64 *v72; // [rsp+10h] [rbp-1E0h]
  size_t v73; // [rsp+10h] [rbp-1E0h]
  _QWORD *src; // [rsp+18h] [rbp-1D8h]
  unsigned __int8 *srca; // [rsp+18h] [rbp-1D8h]
  __int64 v76; // [rsp+20h] [rbp-1D0h]
  _QWORD *v77; // [rsp+20h] [rbp-1D0h]
  unsigned int v78; // [rsp+20h] [rbp-1D0h]
  unsigned int v79; // [rsp+20h] [rbp-1D0h]
  __int64 v80; // [rsp+20h] [rbp-1D0h]
  __int8 *v81; // [rsp+20h] [rbp-1D0h]
  __int64 *v82; // [rsp+20h] [rbp-1D0h]
  size_t n; // [rsp+28h] [rbp-1C8h]
  size_t na; // [rsp+28h] [rbp-1C8h]
  size_t nd; // [rsp+28h] [rbp-1C8h]
  size_t nb; // [rsp+28h] [rbp-1C8h]
  size_t ne; // [rsp+28h] [rbp-1C8h]
  unsigned int nc; // [rsp+28h] [rbp-1C8h]
  void *v90; // [rsp+40h] [rbp-1B0h]
  void *v91; // [rsp+40h] [rbp-1B0h]
  __int8 *v92; // [rsp+40h] [rbp-1B0h]
  void *v93; // [rsp+40h] [rbp-1B0h]
  void *v94; // [rsp+40h] [rbp-1B0h]
  void *v95; // [rsp+40h] [rbp-1B0h]
  void *v96; // [rsp+40h] [rbp-1B0h]
  __int64 v98; // [rsp+50h] [rbp-1A0h]
  _QWORD v99[2]; // [rsp+70h] [rbp-180h] BYREF
  __int16 v100; // [rsp+80h] [rbp-170h]
  unsigned __int64 v101[2]; // [rsp+90h] [rbp-160h] BYREF
  __int16 v102; // [rsp+A0h] [rbp-150h]
  __m128i *p_dest; // [rsp+B0h] [rbp-140h] BYREF
  size_t v104; // [rsp+B8h] [rbp-138h]
  __m128i dest; // [rsp+C0h] [rbp-130h] BYREF

  v10 = (_QWORD *)a2;
  v11 = *(_QWORD *)(a2 + 392);
  if ( *(_QWORD *)(a3 + 8) == v11 && (!v11 || !memcmp(*(const void **)a3, *(const void **)(a2 + 384), v11)) )
  {
    *(_QWORD *)(a3 + 8) = 0;
    *(_QWORD *)a3 = byte_3F871B3;
  }
  if ( !a4[1] )
  {
    a4[1] = 7;
    *a4 = (__int64)"<stdin>";
    *(_QWORD *)a3 = byte_3F871B3;
    *(_QWORD *)(a3 + 8) = 0;
  }
  v12 = *(unsigned int *)(a2 + 128);
  if ( !(_DWORD)v12 )
  {
    *(_BYTE *)(a2 + 489) &= a5 != 0;
    *(_BYTE *)(a2 + 490) |= a5 != 0;
    *(_BYTE *)(a2 + 488) = a6[1].m128i_i8[0];
  }
  if ( !a7 )
  {
    LOBYTE(v98) = 0;
    v104 = 0x10000000000LL;
    p_dest = &dest;
    v99[1] = v98;
    v100 = 2053;
    v101[0] = (unsigned __int64)v99;
    v102 = 1282;
    v99[0] = a3;
    v101[1] = (unsigned __int64)a4;
    sub_16E2F40((__int64)v101, (__int64)&p_dest);
    na = (unsigned int)v104;
    srca = (unsigned __int8 *)p_dest;
    v40 = sub_16D19C0(a2 + 352, (unsigned __int8 *)p_dest, (unsigned int)v104);
    v41 = (__int64 *)(*(_QWORD *)(a2 + 352) + 8LL * v40);
    v42 = *v41;
    if ( *v41 )
    {
      if ( v42 != -8 )
      {
        v43 = p_dest;
        v44 = *(_BYTE *)(a1 + 8) & 0xFC;
        *(_DWORD *)a1 = *(_DWORD *)(v42 + 8);
        *(_BYTE *)(a1 + 8) = v44 | 2;
        if ( v43 != &dest )
          _libc_free((unsigned __int64)v43);
        return a1;
      }
      --*(_DWORD *)(a2 + 368);
    }
    v71 = v41;
    v78 = v40;
    v50 = malloc(na + 17);
    v51 = na;
    v52 = v78;
    v53 = v71;
    v54 = v50;
    if ( !v50 )
    {
      v73 = na;
      v82 = v53;
      nc = v52;
      sub_16BD1C0("Allocation failed", 1u);
      v54 = 0;
      v51 = v73;
      v53 = v82;
      v52 = nc;
    }
    if ( v51 )
    {
      v69 = v54;
      v72 = v53;
      v79 = v52;
      nd = v51;
      memcpy((void *)(v54 + 16), srca, v51);
      v54 = v69;
      v53 = v72;
      v52 = v79;
      v51 = nd;
    }
    v55 = 1;
    *(_BYTE *)(v54 + v51 + 16) = 0;
    if ( (_DWORD)v12 )
      v55 = v12;
    *(_QWORD *)v54 = v51;
    *(_DWORD *)(v54 + 8) = v55;
    *v53 = v54;
    ++*(_DWORD *)(a2 + 364);
    a7 = v55;
    sub_16D1CD0(a2 + 352, v52);
    if ( p_dest != &dest )
      _libc_free((unsigned __int64)p_dest);
    v12 = *(unsigned int *)(a2 + 128);
  }
  v13 = a7;
  if ( a7 < (unsigned int)v12 )
    goto LABEL_8;
  v46 = a7 + 1;
  if ( v46 < v12 )
  {
    v14 = v10[15];
    v47 = 9 * v46;
    v48 = (unsigned __int64 *)(v14 + 72 * v12);
    if ( v48 != (unsigned __int64 *)(v14 + 8 * v47) )
    {
      v77 = v10;
      v49 = v48;
      do
      {
        v49 -= 9;
        if ( (unsigned __int64 *)*v49 != v49 + 2 )
          j_j___libc_free_0(*v49);
      }
      while ( (unsigned __int64 *)(v14 + 8 * v47) != v49 );
      v10 = v77;
      v13 = a7;
      v14 = v77[15];
    }
LABEL_54:
    *((_DWORD *)v10 + 32) = a7 + 1;
    goto LABEL_9;
  }
  if ( v46 > v12 )
  {
    if ( v46 > *((unsigned int *)v10 + 33) )
    {
      sub_38C87C0((__int64)(v10 + 15), v46);
      v12 = *((unsigned int *)v10 + 32);
      v13 = a7;
      v46 = a7 + 1;
    }
    v14 = v10[15];
    v56 = v14 + 72 * v12;
    v57 = v14 + 72 * v46;
    if ( v56 != v57 )
    {
      do
      {
        if ( v56 )
        {
          *(_QWORD *)(v56 + 64) = 0;
          *(_QWORD *)v56 = v56 + 16;
          *(_QWORD *)(v56 + 8) = 0;
          *(_OWORD *)(v56 + 16) = 0;
          *(_OWORD *)(v56 + 32) = 0;
          *(_OWORD *)(v56 + 48) = 0;
        }
        v56 += 72;
      }
      while ( v57 != v56 );
      v14 = v10[15];
    }
    goto LABEL_54;
  }
LABEL_8:
  v14 = v10[15];
LABEL_9:
  v15 = v14 + 72 * v13;
  v16 = *(_QWORD *)(v15 + 8);
  if ( v16 )
  {
    v32 = sub_16BCA90();
    dest.m128i_i8[1] = 1;
    v33 = "file number already allocated";
    v35 = v45;
  }
  else
  {
    if ( *((_BYTE *)v10 + 488) == a6[1].m128i_i8[0] )
    {
      v17 = *(_QWORD *)(a3 + 8);
      if ( v17 )
        goto LABEL_12;
      v91 = *(void **)(v15 + 8);
      v58 = sub_16C40A0(*a4, a4[1], 2);
      v16 = (size_t)v91;
      if ( v13 )
      {
        v80 = v13;
        nb = v58;
        v62 = sub_16C41E0((char *)*a4, a4[1], 2);
        v16 = (size_t)v91;
        *(_QWORD *)(a3 + 8) = v13;
        v63 = *(_QWORD *)(a3 + 8) == 0;
        *(_QWORD *)a3 = v62;
        if ( v63 )
          goto LABEL_75;
        *a4 = nb;
        a4[1] = v80;
      }
      v17 = *(_QWORD *)(a3 + 8);
      if ( v17 )
      {
LABEL_12:
        v18 = *((_DWORD *)v10 + 4);
        v19 = *(__int8 **)a3;
        if ( v18 )
        {
          v76 = a1;
          n = v15;
          v20 = *((_DWORD *)v10 + 4);
          v21 = v10[1];
          src = v10;
          v22 = 0;
          v23 = v17;
          v70 = a4;
          v24 = v19;
          v90 = (void *)v16;
          if ( *(_QWORD *)(v21 + 8) == v17 )
            goto LABEL_16;
          while ( 1 )
          {
            v21 += 32;
            if ( v22 + 1 == v20 )
              break;
            ++v22;
            if ( *(_QWORD *)(v21 + 8) == v23 )
            {
LABEL_16:
              if ( !memcmp(v24, *(const void **)v21, v23) )
              {
                v16 = (size_t)v90;
                v15 = n;
                a1 = v76;
                v10 = src;
                v25 = v22 + 1;
                a4 = v70;
                goto LABEL_18;
              }
            }
          }
          v17 = v23;
          v19 = v24;
          v18 = v20;
          v16 = (size_t)v90;
          v15 = n;
          a1 = v76;
          v10 = src;
          a4 = v70;
          v25 = v22 + 2;
        }
        else
        {
          v25 = 1;
        }
        if ( !v19 )
        {
          dest.m128i_i8[0] = 0;
          p_dest = &dest;
          v104 = 0;
          goto LABEL_95;
        }
        v101[0] = v17;
        p_dest = &dest;
        if ( v17 > 0xF )
        {
          v81 = v19;
          ne = v17;
          v95 = (void *)v16;
          v68 = (unsigned __int8 *)sub_22409D0((__int64)&p_dest, v101, 0);
          v16 = (size_t)v95;
          v17 = ne;
          p_dest = (__m128i *)v68;
          v67 = (__m128i *)v68;
          v19 = v81;
          dest.m128i_i64[0] = v101[0];
        }
        else
        {
          if ( v17 == 1 )
          {
            dest.m128i_i8[0] = *v19;
            v64 = &dest;
LABEL_94:
            v104 = v17;
            v64->m128i_i8[v17] = 0;
            v18 = *((_DWORD *)v10 + 4);
LABEL_95:
            if ( *((_DWORD *)v10 + 5) <= v18 )
            {
              v96 = (void *)v16;
              sub_12BE710((__int64)(v10 + 1), 0, v13, (__int64)v19, v16, v17);
              v18 = *((_DWORD *)v10 + 4);
              v16 = (size_t)v96;
            }
            v65 = (__m128i *)(v10[1] + 32LL * v18);
            if ( v65 )
            {
              v65->m128i_i64[0] = (__int64)v65[1].m128i_i64;
              if ( p_dest == &dest )
              {
                v65[1] = _mm_load_si128(&dest);
              }
              else
              {
                v65->m128i_i64[0] = (__int64)p_dest;
                v65[1].m128i_i64[0] = dest.m128i_i64[0];
              }
              v65->m128i_i64[1] = v104;
              ++*((_DWORD *)v10 + 4);
            }
            else
            {
              v66 = p_dest;
              *((_DWORD *)v10 + 4) = v18 + 1;
              if ( v66 != &dest )
              {
                v93 = (void *)v16;
                j_j___libc_free_0((unsigned __int64)v66);
                v16 = (size_t)v93;
              }
            }
LABEL_18:
            v26 = (__int8 *)*a4;
            if ( *a4 )
            {
LABEL_19:
              v27 = a4[1];
              p_dest = &dest;
              v101[0] = v27;
              if ( v27 > 0xF )
              {
                v92 = v26;
                v60 = (unsigned __int8 *)sub_22409D0((__int64)&p_dest, v101, 0);
                v26 = v92;
                p_dest = (__m128i *)v60;
                v61 = (__m128i *)v60;
                dest.m128i_i64[0] = v101[0];
              }
              else
              {
                if ( v27 == 1 )
                {
                  dest.m128i_i8[0] = *v26;
                  v28 = &dest;
LABEL_22:
                  v104 = v27;
                  v28->m128i_i8[v27] = 0;
                  v29 = *(unsigned __int8 **)v15;
                  v30 = *(__m128i **)v15;
                  if ( p_dest != &dest )
                  {
                    if ( v29 == (unsigned __int8 *)(v15 + 16) )
                    {
                      *(_QWORD *)v15 = p_dest;
                      *(_QWORD *)(v15 + 8) = v104;
                      *(_QWORD *)(v15 + 16) = dest.m128i_i64[0];
                    }
                    else
                    {
                      *(_QWORD *)v15 = p_dest;
                      v31 = *(_QWORD *)(v15 + 16);
                      *(_QWORD *)(v15 + 8) = v104;
                      *(_QWORD *)(v15 + 16) = dest.m128i_i64[0];
                      if ( v29 )
                      {
                        p_dest = (__m128i *)v29;
                        dest.m128i_i64[0] = v31;
                        goto LABEL_26;
                      }
                    }
                    p_dest = &dest;
                    v30 = &dest;
LABEL_26:
                    v104 = 0;
                    v30->m128i_i8[0] = 0;
                    if ( p_dest != &dest )
                      j_j___libc_free_0((unsigned __int64)p_dest);
                    *(_DWORD *)(v15 + 32) = v25;
                    *(_QWORD *)(v15 + 40) = a5;
                    *((_BYTE *)v10 + 489) &= a5 != 0;
                    *((_BYTE *)v10 + 490) |= a5 != 0;
                    if ( a6[1].m128i_i8[0] )
                    {
                      if ( *(_BYTE *)(v15 + 64) )
                      {
                        *(__m128i *)(v15 + 48) = _mm_loadu_si128(a6);
                      }
                      else
                      {
                        v59 = _mm_loadu_si128(a6);
                        *(_BYTE *)(v15 + 64) = 1;
                        *(__m128i *)(v15 + 48) = v59;
                      }
                    }
                    else
                    {
                      if ( !*(_BYTE *)(v15 + 64) )
                      {
LABEL_33:
                        *(_BYTE *)(a1 + 8) = *(_BYTE *)(a1 + 8) & 0xFC | 2;
                        *(_DWORD *)a1 = a7;
                        return a1;
                      }
                      *(_BYTE *)(v15 + 64) = 0;
                    }
                    if ( a6[1].m128i_i8[0] )
                      *((_BYTE *)v10 + 488) = 1;
                    goto LABEL_33;
                  }
                  v16 = v104;
                  if ( v104 )
                  {
                    if ( v104 == 1 )
                      *v29 = dest.m128i_i8[0];
                    else
                      memcpy(v29, &dest, v104);
                    v16 = v104;
                    v29 = *(unsigned __int8 **)v15;
                  }
LABEL_77:
                  *(_QWORD *)(v15 + 8) = v16;
                  v29[v16] = 0;
                  v30 = p_dest;
                  goto LABEL_26;
                }
                if ( !v27 )
                {
                  v28 = &dest;
                  goto LABEL_22;
                }
                v61 = &dest;
              }
              memcpy(v61, v26, v27);
              v27 = v101[0];
              v28 = p_dest;
              goto LABEL_22;
            }
LABEL_76:
            p_dest = &dest;
            v104 = 0;
            dest.m128i_i8[0] = 0;
            v29 = *(unsigned __int8 **)v15;
            goto LABEL_77;
          }
          v67 = &dest;
        }
        v94 = (void *)v16;
        memcpy(v67, v19, v17);
        v17 = v101[0];
        v64 = p_dest;
        v16 = (size_t)v94;
        goto LABEL_94;
      }
LABEL_75:
      v26 = (__int8 *)*a4;
      v25 = 0;
      if ( *a4 )
        goto LABEL_19;
      goto LABEL_76;
    }
    v32 = sub_16BCA90();
    dest.m128i_i8[1] = 1;
    v33 = "inconsistent use of embedded source";
    v35 = v34;
  }
  p_dest = (__m128i *)v33;
  v36 = v32;
  dest.m128i_i8[0] = 3;
  v37 = sub_22077B0(0x38u);
  v38 = v37;
  if ( v37 )
    sub_16BCC70(v37, (__int64)&p_dest, v36, v35);
  *(_BYTE *)(a1 + 8) |= 3u;
  *(_QWORD *)a1 = v38 & 0xFFFFFFFFFFFFFFFELL;
  return a1;
}
