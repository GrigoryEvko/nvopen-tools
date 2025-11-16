// Function: sub_1382970
// Address: 0x1382970
//
unsigned __int64 __fastcall sub_1382970(__int64 a1, __int64 a2)
{
  unsigned __int64 result; // rax
  _DWORD *v3; // r12
  _DWORD *v4; // r9
  _DWORD *v6; // rcx
  _DWORD *v7; // r15
  unsigned int v8; // esi
  __int64 v9; // rbx
  __int64 v10; // rdi
  unsigned int v11; // edx
  _QWORD *v12; // r12
  __int64 v13; // rax
  __m128i *v14; // r13
  __m128i *v15; // r12
  unsigned __int64 v16; // rax
  __m128i *i; // rdx
  unsigned __int64 v18; // rdi
  __int64 v19; // r9
  __m128i *j; // rax
  unsigned __int64 v21; // rsi
  __m128i v22; // xmm0
  __int64 v23; // rax
  __int64 v24; // r14
  __int64 v25; // rbx
  _DWORD *v26; // r10
  __int64 v27; // rdx
  _BYTE *v28; // r11
  signed __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // rdi
  bool v32; // cf
  unsigned __int64 v33; // rax
  __int64 v34; // r15
  __int64 v35; // r15
  __int64 v36; // rax
  char *v37; // r9
  __int64 v38; // r15
  __int64 *v39; // rax
  char *v40; // rax
  __int64 v41; // rsi
  int v42; // r14d
  _QWORD *v43; // r11
  int v44; // eax
  int v45; // edx
  int v46; // eax
  int v47; // esi
  __int64 v48; // rdi
  __int64 v49; // rax
  __int64 v50; // r10
  int v51; // r13d
  _QWORD *v52; // r11
  int v53; // eax
  int v54; // eax
  __int64 v55; // rdi
  _QWORD *v56; // r10
  __int64 v57; // r13
  int v58; // r11d
  __int64 v59; // rsi
  __int64 v60; // [rsp+0h] [rbp-60h]
  _DWORD *v61; // [rsp+8h] [rbp-58h]
  _DWORD *v62; // [rsp+10h] [rbp-50h]
  __int64 v63; // [rsp+10h] [rbp-50h]
  __int64 v64; // [rsp+10h] [rbp-50h]
  size_t n; // [rsp+18h] [rbp-48h]
  size_t na; // [rsp+18h] [rbp-48h]
  size_t nb; // [rsp+18h] [rbp-48h]
  void *src; // [rsp+20h] [rbp-40h]
  _BYTE *srcc; // [rsp+20h] [rbp-40h]
  _DWORD *srcd; // [rsp+20h] [rbp-40h]
  _DWORD *srce; // [rsp+20h] [rbp-40h]
  _DWORD *srca; // [rsp+20h] [rbp-40h]
  _DWORD *srcb; // [rsp+20h] [rbp-40h]
  _DWORD *v74; // [rsp+28h] [rbp-38h]
  __int64 v75; // [rsp+28h] [rbp-38h]
  _BYTE *v76; // [rsp+28h] [rbp-38h]
  char *v77; // [rsp+28h] [rbp-38h]
  __int64 v78; // [rsp+28h] [rbp-38h]
  __int64 v79; // [rsp+28h] [rbp-38h]

  result = *(unsigned int *)(a2 + 24);
  v3 = *(_DWORD **)(a2 + 8);
  v4 = &v3[12 * result];
  if ( *(_DWORD *)(a2 + 16) && v3 != v4 )
  {
    while ( 1 )
    {
      result = *(_QWORD *)v3;
      if ( *(_QWORD *)v3 != -8 )
        break;
      if ( v3[2] != -1 )
        goto LABEL_6;
LABEL_75:
      v3 += 12;
      if ( v4 == v3 )
        return result;
    }
    if ( result == -16 && v3[2] == -2 )
      goto LABEL_75;
LABEL_6:
    if ( v4 != v3 )
    {
      v6 = v3;
      v7 = v4;
      do
      {
        if ( v6[2] )
          goto LABEL_11;
        v8 = *(_DWORD *)(a1 + 24);
        v9 = *(_QWORD *)v6;
        if ( v8 )
        {
          v10 = *(_QWORD *)(a1 + 8);
          v11 = (v8 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
          v12 = (_QWORD *)(v10 + 32LL * v11);
          v13 = *v12;
          if ( v9 == *v12 )
            goto LABEL_20;
          v42 = 1;
          v43 = 0;
          while ( v13 != -8 )
          {
            if ( v13 == -16 && !v43 )
              v43 = v12;
            v11 = (v8 - 1) & (v42 + v11);
            v12 = (_QWORD *)(v10 + 32LL * v11);
            v13 = *v12;
            if ( v9 == *v12 )
              goto LABEL_20;
            ++v42;
          }
          v44 = *(_DWORD *)(a1 + 16);
          if ( v43 )
            v12 = v43;
          ++*(_QWORD *)a1;
          v45 = v44 + 1;
          if ( 4 * (v44 + 1) < 3 * v8 )
          {
            if ( v8 - *(_DWORD *)(a1 + 20) - v45 <= v8 >> 3 )
            {
              v79 = a1;
              srcb = v6;
              sub_13824C0(a1, v8);
              a1 = v79;
              v53 = *(_DWORD *)(v79 + 24);
              if ( !v53 )
              {
LABEL_120:
                ++*(_DWORD *)(a1 + 16);
                BUG();
              }
              v54 = v53 - 1;
              v55 = *(_QWORD *)(v79 + 8);
              v56 = 0;
              LODWORD(v57) = v54 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
              v6 = srcb;
              v58 = 1;
              v45 = *(_DWORD *)(v79 + 16) + 1;
              v12 = (_QWORD *)(v55 + 32LL * (unsigned int)v57);
              v59 = *v12;
              if ( v9 != *v12 )
              {
                while ( v59 != -8 )
                {
                  if ( v59 == -16 && !v56 )
                    v56 = v12;
                  v57 = v54 & (unsigned int)(v57 + v58);
                  v12 = (_QWORD *)(v55 + 32 * v57);
                  v59 = *v12;
                  if ( v9 == *v12 )
                    goto LABEL_83;
                  ++v58;
                }
                if ( v56 )
                  v12 = v56;
              }
            }
            goto LABEL_83;
          }
        }
        else
        {
          ++*(_QWORD *)a1;
        }
        v78 = a1;
        srca = v6;
        sub_13824C0(a1, 2 * v8);
        a1 = v78;
        v46 = *(_DWORD *)(v78 + 24);
        if ( !v46 )
          goto LABEL_120;
        v47 = v46 - 1;
        v48 = *(_QWORD *)(v78 + 8);
        v6 = srca;
        LODWORD(v49) = (v46 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v45 = *(_DWORD *)(v78 + 16) + 1;
        v12 = (_QWORD *)(v48 + 32LL * (unsigned int)v49);
        v50 = *v12;
        if ( v9 != *v12 )
        {
          v51 = 1;
          v52 = 0;
          while ( v50 != -8 )
          {
            if ( !v52 && v50 == -16 )
              v52 = v12;
            v49 = v47 & (unsigned int)(v49 + v51);
            v12 = (_QWORD *)(v48 + 32 * v49);
            v50 = *v12;
            if ( v9 == *v12 )
              goto LABEL_83;
            ++v51;
          }
          if ( v52 )
            v12 = v52;
        }
LABEL_83:
        *(_DWORD *)(a1 + 16) = v45;
        if ( *v12 != -8 )
          --*(_DWORD *)(a1 + 20);
        *v12 = v9;
        v12[1] = 0;
        v12[2] = 0;
        v12[3] = 0;
LABEL_20:
        if ( !v6[8] || (v23 = *((_QWORD *)v6 + 3), v24 = v23 + 24LL * (unsigned int)v6[10], v23 == v24) )
        {
LABEL_21:
          v14 = (__m128i *)v12[2];
        }
        else
        {
          while ( 1 )
          {
            v25 = v23;
            if ( *(_QWORD *)v23 == -8 )
            {
              if ( *(_DWORD *)(v23 + 8) != -1 )
                break;
              goto LABEL_70;
            }
            if ( *(_QWORD *)v23 != -16 || *(_DWORD *)(v23 + 8) != -2 )
              break;
LABEL_70:
            v23 += 24;
            if ( v24 == v23 )
              goto LABEL_21;
          }
          v14 = (__m128i *)v12[2];
          if ( v23 != v24 )
          {
            v26 = v7;
            while ( 1 )
            {
              if ( *(_DWORD *)(v25 + 8) )
                goto LABEL_40;
              v27 = *(_QWORD *)v25;
              if ( v14 == (__m128i *)v12[3] )
                break;
              if ( v14 )
              {
                v14->m128i_i64[0] = v27;
                v14->m128i_i64[1] = 0x7FFFFFFFFFFFFFFFLL;
                v14 = (__m128i *)v12[2];
              }
              v12[2] = ++v14;
LABEL_40:
              v25 += 24;
              if ( v25 == v24 )
              {
LABEL_44:
                v7 = v26;
                goto LABEL_22;
              }
              while ( *(_QWORD *)v25 == -8 )
              {
                if ( *(_DWORD *)(v25 + 8) != -1 )
                  goto LABEL_38;
LABEL_43:
                v25 += 24;
                if ( v24 == v25 )
                  goto LABEL_44;
              }
              if ( *(_QWORD *)v25 == -16 && *(_DWORD *)(v25 + 8) == -2 )
                goto LABEL_43;
LABEL_38:
              if ( v24 == v25 )
                goto LABEL_44;
            }
            v28 = (_BYTE *)v12[1];
            v29 = (char *)v14 - v28;
            v30 = ((char *)v14 - v28) >> 4;
            if ( v30 == 0x7FFFFFFFFFFFFFFLL )
              sub_4262D8((__int64)"vector::_M_realloc_insert");
            v31 = 1;
            if ( v30 )
              v31 = ((char *)v14 - v28) >> 4;
            v32 = __CFADD__(v31, v30);
            v33 = v31 + v30;
            if ( v32 )
            {
              v35 = 0x7FFFFFFFFFFFFFF0LL;
              goto LABEL_59;
            }
            if ( v33 )
            {
              v34 = 0x7FFFFFFFFFFFFFFLL;
              if ( v33 <= 0x7FFFFFFFFFFFFFFLL )
                v34 = v33;
              v35 = 16 * v34;
LABEL_59:
              v60 = a1;
              v61 = v6;
              v62 = v26;
              n = (char *)v14 - v28;
              srcc = (_BYTE *)v12[1];
              v75 = *(_QWORD *)v25;
              v36 = sub_22077B0(v35);
              v27 = v75;
              v28 = srcc;
              v29 = n;
              v26 = v62;
              v37 = (char *)v36;
              v38 = v36 + v35;
              v6 = v61;
              a1 = v60;
            }
            else
            {
              v38 = 0;
              v37 = 0;
            }
            v39 = (__int64 *)&v37[v29];
            if ( &v37[v29] )
            {
              *v39 = v27;
              v39[1] = 0x7FFFFFFFFFFFFFFFLL;
            }
            v14 = (__m128i *)&v37[v29 + 16];
            if ( v29 > 0 )
            {
              v63 = a1;
              na = (size_t)v6;
              srcd = v26;
              v76 = v28;
              v40 = (char *)memmove(v37, v28, v29);
              v28 = v76;
              v26 = srcd;
              v6 = (_DWORD *)na;
              v37 = v40;
              a1 = v63;
              v41 = v12[3] - (_QWORD)v76;
            }
            else
            {
              if ( !v28 )
              {
LABEL_64:
                v12[1] = v37;
                v12[2] = v14;
                v12[3] = v38;
                goto LABEL_40;
              }
              v41 = v12[3] - (_QWORD)v28;
            }
            v64 = a1;
            nb = (size_t)v6;
            srce = v26;
            v77 = v37;
            j_j___libc_free_0(v28, v41);
            a1 = v64;
            v6 = (_DWORD *)nb;
            v26 = srce;
            v37 = v77;
            goto LABEL_64;
          }
        }
LABEL_22:
        v15 = (__m128i *)v12[1];
        if ( v15 != v14 )
        {
          src = (void *)a1;
          v74 = v6;
          _BitScanReverse64(&v16, v14 - v15);
          sub_1381B80(v15, v14, 2LL * (int)(63 - (v16 ^ 0x3F)));
          if ( (char *)v14 - (char *)v15 <= 256 )
          {
            sub_1381E40(v15->m128i_i8, v14->m128i_i8);
            a1 = (__int64)src;
            v6 = v74;
          }
          else
          {
            sub_1381E40(v15->m128i_i8, v15[16].m128i_i8);
            v6 = v74;
            a1 = (__int64)src;
            for ( i = v15 + 16; i != v14; j->m128i_i64[1] = v19 )
            {
              v18 = i->m128i_i64[0];
              v19 = i->m128i_i64[1];
              for ( j = i; ; j[1] = v22 )
              {
                v21 = j[-1].m128i_u64[0];
                if ( v21 <= v18 && (v19 >= j[-1].m128i_i64[1] || v18 != v21) )
                  break;
                v22 = _mm_loadu_si128(--j);
              }
              ++i;
              j->m128i_i64[0] = v18;
            }
          }
        }
LABEL_11:
        result = (unsigned __int64)(v6 + 12);
        if ( v6 + 12 == v7 )
          return result;
        while ( 1 )
        {
          v6 = (_DWORD *)result;
          if ( *(_QWORD *)result == -8 )
          {
            if ( *(_DWORD *)(result + 8) != -1 )
              break;
            goto LABEL_14;
          }
          if ( *(_QWORD *)result != -16 || *(_DWORD *)(result + 8) != -2 )
            break;
LABEL_14:
          result += 48LL;
          if ( v7 == (_DWORD *)result )
            return result;
        }
      }
      while ( v7 != (_DWORD *)result );
    }
  }
  return result;
}
