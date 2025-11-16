// Function: sub_24D75B0
// Address: 0x24d75b0
//
void __fastcall sub_24D75B0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // rbx
  __int64 i; // r14
  __int64 v9; // rax
  char v10; // al
  __int64 v11; // r12
  __int64 v12; // rax
  int v13; // eax
  _QWORD *v14; // rax
  __int64 v15; // rcx
  __int64 *v16; // rsi
  __int64 v17; // rdi
  _QWORD *v18; // rdx
  __m128i *v19; // r12
  unsigned __int64 v20; // rcx
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __m128i *v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 *v27; // r15
  __int64 *v28; // rbx
  unsigned int v29; // edx
  __int64 *v30; // rax
  __int64 v31; // rdi
  unsigned int v32; // esi
  int v33; // eax
  __int64 v34; // r11
  int v35; // edx
  unsigned int v36; // ecx
  __int64 v37; // rdi
  unsigned int v38; // esi
  __int64 v39; // r10
  _QWORD *v40; // rdx
  __int64 v41; // rcx
  int v42; // eax
  __int64 v43; // rax
  const void *v44; // rsi
  __int8 *v45; // r12
  int v46; // r11d
  __int64 *v47; // r10
  int v48; // ecx
  int v49; // eax
  __int64 v50; // r11
  int v51; // r10d
  __int64 *v52; // rsi
  unsigned int v53; // ecx
  __int64 v54; // rdi
  int v55; // eax
  int v56; // edi
  __int64 v57; // r10
  __int64 v58; // rsi
  _QWORD *v59; // rcx
  int v60; // r10d
  int v61; // r10d
  unsigned int v62; // edi
  __int64 v63; // rsi
  int v64; // r10d
  __int64 v65; // [rsp+10h] [rbp-E0h]
  int v66; // [rsp+18h] [rbp-D8h]
  __int64 v67; // [rsp+20h] [rbp-D0h]
  int v68; // [rsp+20h] [rbp-D0h]
  const void *v70; // [rsp+30h] [rbp-C0h]
  __int64 v73; // [rsp+48h] [rbp-A8h]
  __int64 v74; // [rsp+50h] [rbp-A0h]
  __int64 v75; // [rsp+58h] [rbp-98h]
  __int64 v76; // [rsp+60h] [rbp-90h]
  unsigned int v77; // [rsp+68h] [rbp-88h]
  __int64 v78; // [rsp+70h] [rbp-80h]
  __int64 v79; // [rsp+78h] [rbp-78h]
  __m128i v80; // [rsp+80h] [rbp-70h] BYREF
  __int64 v81; // [rsp+90h] [rbp-60h]
  __int64 v82; // [rsp+98h] [rbp-58h]
  __int64 v83; // [rsp+A0h] [rbp-50h]
  __int64 v84; // [rsp+A8h] [rbp-48h]
  __int64 v85; // [rsp+B0h] [rbp-40h]

  v6 = a1 + 72;
  v7 = *(_QWORD *)(a1 + 80);
  v78 = a5;
  if ( a1 + 72 == v7 )
  {
    i = 0;
  }
  else
  {
    if ( !v7 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v7 + 32);
      if ( i != v7 + 24 )
        break;
      v7 = *(_QWORD *)(v7 + 8);
      if ( v6 == v7 )
        return;
      if ( !v7 )
        BUG();
    }
  }
  if ( v6 != v7 )
  {
    v70 = (const void *)(a5 + 16);
    while ( 1 )
    {
      if ( !i )
        BUG();
      if ( (*(_BYTE *)(i - 17) & 0x20) != 0 && sub_B91C10(i - 24, 31) )
        goto LABEL_14;
      v10 = *(_BYTE *)(i - 24);
      if ( ((v10 - 61) & 0xFA) == 0 )
        break;
      switch ( v10 )
      {
        case '"':
          goto LABEL_14;
        case 'U':
          sub_F58670(i - 24, a2);
          if ( *(_BYTE *)(i - 24) != 85 )
            goto LABEL_14;
          v25 = *(_QWORD *)(i - 56);
          if ( !v25 )
            goto LABEL_14;
          if ( (*(_BYTE *)v25
             || *(_QWORD *)(v25 + 24) != *(_QWORD *)(i + 56)
             || (*(_BYTE *)(v25 + 33) & 0x20) == 0
             || (unsigned int)(*(_DWORD *)(v25 + 36) - 238) > 7
             || ((1LL << (*(_BYTE *)(v25 + 36) + 18)) & 0xAD) == 0)
            && (*(_BYTE *)v25
             || *(_QWORD *)(v25 + 24) != *(_QWORD *)(i + 56)
             || (*(_BYTE *)(v25 + 33) & 0x20) == 0
             || (unsigned int)(*(_DWORD *)(v25 + 36) - 210) > 1) )
          {
            goto LABEL_14;
          }
          break;
        case '<':
          break;
        default:
          goto LABEL_14;
      }
      v24 = *(unsigned int *)(v78 + 8);
      if ( v24 + 1 > (unsigned __int64)*(unsigned int *)(v78 + 12) )
      {
        sub_C8D5F0(v78, v70, v24 + 1, 8u, a5, a6);
        v24 = *(unsigned int *)(v78 + 8);
      }
      *(_QWORD *)(*(_QWORD *)v78 + 8 * v24) = i - 24;
      ++*(_DWORD *)(v78 + 8);
LABEL_14:
      for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v7 + 32) )
      {
        v9 = v7 - 24;
        if ( !v7 )
          v9 = 0;
        if ( i != v9 + 48 )
          break;
        v7 = *(_QWORD *)(v7 + 8);
        if ( v6 == v7 )
          return;
        if ( !v7 )
          BUG();
      }
      if ( v6 == v7 )
        return;
    }
    sub_D66840(&v80, (_BYTE *)(i - 24));
    v73 = v80.m128i_i64[1];
    v74 = v84;
    v11 = v81;
    v76 = v82;
    v75 = v83;
    v79 = v80.m128i_i64[0];
    if ( (unsigned __int8)sub_BD6020(v80.m128i_i64[0]) )
      goto LABEL_14;
    v12 = *(_QWORD *)(v79 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v12 + 8) - 17 <= 1 )
      v12 = **(_QWORD **)(v12 + 16);
    if ( *(_DWORD *)(v12 + 8) >> 8 )
      goto LABEL_14;
    if ( !v11 )
      goto LABEL_38;
    v13 = *(_DWORD *)(a4 + 16);
    if ( !v13 )
    {
      v14 = *(_QWORD **)(a4 + 32);
      v15 = *(unsigned int *)(a4 + 40);
      v16 = &v14[v15];
      v17 = (8 * v15) >> 3;
      if ( (8 * v15) >> 5 )
      {
        v18 = &v14[4 * ((8 * v15) >> 5)];
        while ( *v14 != v11 )
        {
          if ( v14[1] == v11 )
          {
            ++v14;
            break;
          }
          if ( v14[2] == v11 )
          {
            v14 += 2;
            break;
          }
          if ( v14[3] == v11 )
          {
            v14 += 3;
            break;
          }
          v14 += 4;
          if ( v18 == v14 )
          {
            v17 = v16 - v14;
            goto LABEL_57;
          }
        }
LABEL_37:
        if ( v16 != v14 )
          goto LABEL_38;
LABEL_60:
        if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 44) )
        {
          sub_C8D5F0(a4 + 32, (const void *)(a4 + 48), v15 + 1, 8u, a5, a6);
          v16 = (__int64 *)(*(_QWORD *)(a4 + 32) + 8LL * *(unsigned int *)(a4 + 40));
        }
        *v16 = v11;
        v26 = (unsigned int)(*(_DWORD *)(a4 + 40) + 1);
        *(_DWORD *)(a4 + 40) = v26;
        if ( (unsigned int)v26 > 8 )
        {
          v67 = v6;
          v65 = v7;
          v27 = *(__int64 **)(a4 + 32);
          v28 = &v27[v26];
          while ( 1 )
          {
            v32 = *(_DWORD *)(a4 + 24);
            if ( !v32 )
              break;
            a6 = v32 - 1;
            a5 = *(_QWORD *)(a4 + 8);
            v29 = a6 & (((unsigned int)*v27 >> 9) ^ ((unsigned int)*v27 >> 4));
            v30 = (__int64 *)(a5 + 8LL * v29);
            v31 = *v30;
            if ( *v27 != *v30 )
            {
              v46 = 1;
              v47 = 0;
              while ( v31 != -4096 )
              {
                if ( v31 == -8192 && !v47 )
                  v47 = v30;
                v29 = a6 & (v46 + v29);
                v30 = (__int64 *)(a5 + 8LL * v29);
                v31 = *v30;
                if ( *v27 == *v30 )
                  goto LABEL_65;
                ++v46;
              }
              v48 = *(_DWORD *)(a4 + 16);
              if ( v47 )
                v30 = v47;
              ++*(_QWORD *)a4;
              v35 = v48 + 1;
              if ( 4 * (v48 + 1) < 3 * v32 )
              {
                if ( v32 - *(_DWORD *)(a4 + 20) - v35 <= v32 >> 3 )
                {
                  sub_BD1680(a4, v32);
                  v49 = *(_DWORD *)(a4 + 24);
                  if ( !v49 )
                  {
LABEL_159:
                    ++*(_DWORD *)(a4 + 16);
                    BUG();
                  }
                  a5 = *v27;
                  a6 = (unsigned int)(v49 - 1);
                  v50 = *(_QWORD *)(a4 + 8);
                  v51 = 1;
                  v35 = *(_DWORD *)(a4 + 16) + 1;
                  v52 = 0;
                  v53 = a6 & (((unsigned int)*v27 >> 9) ^ ((unsigned int)*v27 >> 4));
                  v30 = (__int64 *)(v50 + 8LL * v53);
                  v54 = *v30;
                  if ( *v27 != *v30 )
                  {
                    while ( v54 != -4096 )
                    {
                      if ( v54 == -8192 && !v52 )
                        v52 = v30;
                      v53 = a6 & (v51 + v53);
                      v30 = (__int64 *)(v50 + 8LL * v53);
                      v54 = *v30;
                      if ( a5 == *v30 )
                        goto LABEL_70;
                      ++v51;
                    }
LABEL_109:
                    if ( v52 )
                      v30 = v52;
                  }
                }
LABEL_70:
                *(_DWORD *)(a4 + 16) = v35;
                if ( *v30 != -4096 )
                  --*(_DWORD *)(a4 + 20);
                *v30 = *v27;
                goto LABEL_65;
              }
LABEL_68:
              sub_BD1680(a4, 2 * v32);
              v33 = *(_DWORD *)(a4 + 24);
              if ( !v33 )
                goto LABEL_159;
              a5 = *v27;
              a6 = (unsigned int)(v33 - 1);
              v34 = *(_QWORD *)(a4 + 8);
              v35 = *(_DWORD *)(a4 + 16) + 1;
              v36 = a6 & (((unsigned int)*v27 >> 9) ^ ((unsigned int)*v27 >> 4));
              v30 = (__int64 *)(v34 + 8LL * v36);
              v37 = *v30;
              if ( *v27 != *v30 )
              {
                v64 = 1;
                v52 = 0;
                while ( v37 != -4096 )
                {
                  if ( v37 == -8192 && !v52 )
                    v52 = v30;
                  v36 = a6 & (v64 + v36);
                  v30 = (__int64 *)(v34 + 8LL * v36);
                  v37 = *v30;
                  if ( a5 == *v30 )
                    goto LABEL_70;
                  ++v64;
                }
                goto LABEL_109;
              }
              goto LABEL_70;
            }
LABEL_65:
            if ( v28 == ++v27 )
            {
              v6 = v67;
              v7 = v65;
              goto LABEL_38;
            }
          }
          ++*(_QWORD *)a4;
          goto LABEL_68;
        }
LABEL_38:
        v82 = v11;
        v19 = &v80;
        v80.m128i_i64[0] = i - 24;
        v80.m128i_i64[1] = v79;
        v20 = *(_QWORD *)a3;
        v81 = v73;
        v83 = v76;
        v84 = v75;
        v85 = v74;
        v21 = *(unsigned int *)(a3 + 8);
        v22 = v21 + 1;
        if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
        {
          v44 = (const void *)(a3 + 16);
          if ( v20 > (unsigned __int64)&v80 || (unsigned __int64)&v80 >= v20 + 56 * v21 )
          {
            sub_C8D5F0(a3, v44, v22, 0x38u, a5, a6);
            v20 = *(_QWORD *)a3;
            v21 = *(unsigned int *)(a3 + 8);
            v19 = &v80;
          }
          else
          {
            v45 = &v80.m128i_i8[-v20];
            sub_C8D5F0(a3, v44, v22, 0x38u, a5, a6);
            v20 = *(_QWORD *)a3;
            v21 = *(unsigned int *)(a3 + 8);
            v19 = (__m128i *)&v45[*(_QWORD *)a3];
          }
        }
        v23 = (__m128i *)(v20 + 56 * v21);
        *v23 = _mm_loadu_si128(v19);
        v23[1] = _mm_loadu_si128(v19 + 1);
        v23[2] = _mm_loadu_si128(v19 + 2);
        v23[3].m128i_i64[0] = v19[3].m128i_i64[0];
        ++*(_DWORD *)(a3 + 8);
        goto LABEL_14;
      }
LABEL_57:
      if ( v17 != 2 )
      {
        if ( v17 != 3 )
        {
          if ( v17 != 1 )
            goto LABEL_60;
          goto LABEL_97;
        }
        if ( *v14 == v11 )
          goto LABEL_37;
        ++v14;
      }
      if ( *v14 == v11 )
        goto LABEL_37;
      ++v14;
LABEL_97:
      if ( *v14 == v11 )
        goto LABEL_37;
      goto LABEL_60;
    }
    v38 = *(_DWORD *)(a4 + 24);
    if ( v38 )
    {
      v39 = *(_QWORD *)(a4 + 8);
      v77 = ((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4);
      a5 = (v38 - 1) & v77;
      v40 = (_QWORD *)(v39 + 8 * a5);
      v41 = *v40;
      if ( v11 == *v40 )
        goto LABEL_38;
      v68 = 1;
      a6 = 0;
      v66 = *(_DWORD *)(a4 + 16);
      while ( v41 != -4096 )
      {
        if ( v41 == -8192 && !a6 )
          a6 = (__int64)v40;
        a5 = (v38 - 1) & (v68 + (_DWORD)a5);
        v40 = (_QWORD *)(v39 + 8LL * (unsigned int)a5);
        v41 = *v40;
        if ( v11 == *v40 )
          goto LABEL_38;
        ++v68;
      }
      if ( a6 )
        v40 = (_QWORD *)a6;
      v42 = v13 + 1;
      ++*(_QWORD *)a4;
      if ( 4 * (v66 + 1) < 3 * v38 )
      {
        if ( v38 - *(_DWORD *)(a4 + 20) - v42 > v38 >> 3 )
          goto LABEL_81;
        sub_BD1680(a4, v38);
        v60 = *(_DWORD *)(a4 + 24);
        if ( !v60 )
        {
LABEL_160:
          ++*(_DWORD *)(a4 + 16);
          BUG();
        }
        v61 = v60 - 1;
        a6 = *(_QWORD *)(a4 + 8);
        a5 = 1;
        v62 = v61 & v77;
        v40 = (_QWORD *)(a6 + 8LL * (v61 & v77));
        v59 = 0;
        v63 = *v40;
        v42 = *(_DWORD *)(a4 + 16) + 1;
        if ( v11 == *v40 )
          goto LABEL_81;
        while ( v63 != -4096 )
        {
          if ( v63 == -8192 && !v59 )
            v59 = v40;
          v62 = v61 & (a5 + v62);
          v40 = (_QWORD *)(a6 + 8LL * v62);
          v63 = *v40;
          if ( v11 == *v40 )
            goto LABEL_81;
          a5 = (unsigned int)(a5 + 1);
        }
        goto LABEL_121;
      }
    }
    else
    {
      ++*(_QWORD *)a4;
    }
    sub_BD1680(a4, 2 * v38);
    v55 = *(_DWORD *)(a4 + 24);
    if ( !v55 )
      goto LABEL_160;
    v56 = v55 - 1;
    v57 = *(_QWORD *)(a4 + 8);
    a5 = (v55 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
    v40 = (_QWORD *)(v57 + 8 * a5);
    v58 = *v40;
    v42 = *(_DWORD *)(a4 + 16) + 1;
    if ( v11 == *v40 )
      goto LABEL_81;
    a6 = 1;
    v59 = 0;
    while ( v58 != -4096 )
    {
      if ( !v59 && v58 == -8192 )
        v59 = v40;
      a5 = v56 & (unsigned int)(a6 + a5);
      v40 = (_QWORD *)(v57 + 8LL * (unsigned int)a5);
      v58 = *v40;
      if ( v11 == *v40 )
        goto LABEL_81;
      a6 = (unsigned int)(a6 + 1);
    }
LABEL_121:
    if ( v59 )
      v40 = v59;
LABEL_81:
    *(_DWORD *)(a4 + 16) = v42;
    if ( *v40 != -4096 )
      --*(_DWORD *)(a4 + 20);
    *v40 = v11;
    v43 = *(unsigned int *)(a4 + 40);
    if ( v43 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 44) )
    {
      sub_C8D5F0(a4 + 32, (const void *)(a4 + 48), v43 + 1, 8u, a5, a6);
      v43 = *(unsigned int *)(a4 + 40);
    }
    *(_QWORD *)(*(_QWORD *)(a4 + 32) + 8 * v43) = v11;
    ++*(_DWORD *)(a4 + 40);
    goto LABEL_38;
  }
}
