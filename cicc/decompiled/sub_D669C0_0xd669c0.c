// Function: sub_D669C0
// Address: 0xd669c0
//
__m128i *__fastcall sub_D669C0(__m128i *a1, __int64 a2, unsigned int a3, __int64 *a4)
{
  __int64 v6; // r13
  char v9; // al
  _QWORD *v10; // r9
  char v11; // al
  __int64 v12; // rsi
  int v13; // eax
  __m128i v14; // xmm0
  __m128i v15; // xmm1
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  unsigned int v20; // eax
  __int64 v21; // rdx
  unsigned __int64 v22; // rax
  __m128i v23; // xmm6
  __m128i v24; // xmm7
  __int64 v25; // rdx
  unsigned __int64 v26; // rax
  unsigned int v27; // ecx
  __m128i v28; // xmm4
  __m128i v29; // xmm5
  __int64 v30; // rdx
  unsigned __int64 v31; // rax
  __m128i v32; // xmm2
  __m128i v33; // xmm3
  __m128i v34; // xmm2
  __m128i v35; // xmm3
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rbx
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rbx
  __int64 v43; // rdx
  __int64 v44; // rdx
  __int64 v45; // r12
  __int64 v46; // r14
  char v47; // bl
  __int64 v48; // rdx
  char v49; // si
  _QWORD *v50; // rax
  unsigned __int64 v51; // rax
  unsigned __int64 v52; // rdx
  __int64 v53; // rdx
  __int64 v54; // rdx
  __int64 v55; // rdx
  __int64 v56; // rdx
  unsigned __int64 v57; // rdx
  bool v58; // cc
  __int64 v59; // rdx
  unsigned int v60; // edx
  unsigned __int64 v61; // rdx
  __int64 v62; // [rsp+8h] [rbp-68h]
  __int64 v63; // [rsp+10h] [rbp-60h] BYREF
  __int64 v64; // [rsp+18h] [rbp-58h]
  __m128i v65; // [rsp+20h] [rbp-50h] BYREF
  __m128i v66[4]; // [rsp+30h] [rbp-40h] BYREF

  v6 = a3;
  sub_B91FC0(v65.m128i_i64, a2);
  v62 = *(_QWORD *)(a2 + 32 * (v6 - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( *(_BYTE *)a2 == 85 )
  {
    v17 = *(_QWORD *)(a2 - 32);
    if ( v17 )
    {
      if ( !*(_BYTE *)v17 && *(_QWORD *)(v17 + 24) == *(_QWORD *)(a2 + 80) && (*(_BYTE *)(v17 + 33) & 0x20) != 0 )
      {
        v18 = sub_B43CC0(a2);
        v19 = *(_QWORD *)(a2 - 32);
        if ( !v19 || *(_BYTE *)v19 || *(_QWORD *)(v19 + 24) != *(_QWORD *)(a2 + 80) )
          BUG();
        v20 = *(_DWORD *)(v19 + 36);
        if ( v20 > 0xF4 )
        {
          if ( v20 == 3644 )
          {
            v63 = sub_9208B0(v18, *(_QWORD *)(a2 + 8));
            v64 = v54;
            v31 = (unsigned __int64)(v63 + 7) >> 3;
            if ( (_BYTE)v54 )
              v31 |= 0x4000000000000000uLL;
LABEL_49:
            a1->m128i_i64[0] = v62;
            goto LABEL_50;
          }
          if ( v20 == 3717 )
          {
            v63 = sub_9208B0(v18, *(_QWORD *)(*(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) + 8LL));
            v64 = v53;
            v26 = (unsigned __int64)(v63 + 7) >> 3;
            if ( (_BYTE)v53 )
              v26 |= 0x4000000000000000uLL;
            goto LABEL_43;
          }
        }
        else
        {
          if ( v20 > 0xCB )
          {
            switch ( v20 )
            {
              case 0xCCu:
                if ( !(_DWORD)v6 )
                {
                  v34 = _mm_loadu_si128(&v65);
                  a1->m128i_i64[1] = 0;
                  v35 = _mm_loadu_si128(v66);
                  a1->m128i_i64[0] = v62;
                  a1[1] = v34;
                  a1[2] = v35;
                  return a1;
                }
                v59 = *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
                v26 = *(_QWORD *)(v59 + 24);
                if ( *(_DWORD *)(v59 + 32) <= 0x40u )
                  goto LABEL_41;
                goto LABEL_40;
              case 0xCDu:
              case 0xD2u:
              case 0xD3u:
                v30 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
                v31 = *(_QWORD *)(v30 + 24);
                if ( *(_DWORD *)(v30 + 32) > 0x40u )
                  v31 = *(_QWORD *)v31;
                if ( v31 > 0x3FFFFFFFFFFFFFFBLL )
                  v31 = 0xBFFFFFFFFFFFFFFELL;
                goto LABEL_49;
              case 0xE4u:
                v40 = sub_9208B0(v18, *(_QWORD *)(a2 + 8));
                v42 = v41;
                v63 = v40;
                v43 = v40;
                v26 = 0xBFFFFFFFFFFFFFFELL;
                v64 = v42;
                if ( !(_BYTE)v42 )
                {
                  v26 = (unsigned __int64)(v43 + 7) >> 3;
                  if ( v26 )
                    v26 |= 0x8000000000000000LL;
                }
                goto LABEL_43;
              case 0xE6u:
                v36 = sub_9208B0(v18, *(_QWORD *)(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) + 8LL));
                v38 = v37;
                v63 = v36;
                v39 = v36;
                v22 = 0xBFFFFFFFFFFFFFFELL;
                v64 = v38;
                if ( !(_BYTE)v38 )
                {
                  v22 = (unsigned __int64)(v39 + 7) >> 3;
                  if ( v22 )
                    v22 |= 0x8000000000000000LL;
                }
                goto LABEL_56;
              case 0xEEu:
              case 0xEFu:
              case 0xF0u:
              case 0xF1u:
              case 0xF2u:
              case 0xF3u:
              case 0xF4u:
                goto LABEL_23;
              default:
                goto LABEL_2;
            }
          }
          if ( v20 == 154 )
          {
            v44 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
            v45 = *(_QWORD *)(a2 + 32 * (2 - v44));
            if ( *(_BYTE *)v45 == 17 )
            {
              v46 = *(_QWORD *)(*(_QWORD *)(a2 + 32 * (1 - v44)) + 8LL);
              v47 = sub_AE5020(v18, v46);
              v63 = sub_9208B0(v18, v46);
              v64 = v48;
              v49 = v48;
              v50 = *(_QWORD **)(v45 + 24);
              if ( *(_DWORD *)(v45 + 32) > 0x40u )
                v50 = (_QWORD *)*v50;
              v51 = ((((unsigned __int64)(v63 + 7) >> 3) + (1LL << v47) - 1) >> v47 << v47) * (_QWORD)v50;
              if ( v51 > 0x3FFFFFFFFFFFFFFBLL )
              {
                v22 = 0xBFFFFFFFFFFFFFFELL;
              }
              else
              {
                v52 = v51;
                v22 = v51 | 0x4000000000000000LL;
                if ( !v49 )
                  v22 = v52;
              }
              goto LABEL_56;
            }
            a1->m128i_i64[0] = v62;
            v31 = 0xBFFFFFFFFFFFFFFELL;
LABEL_50:
            v32 = _mm_loadu_si128(&v65);
            v33 = _mm_loadu_si128(v66);
            a1->m128i_i64[1] = v31;
            a1[1] = v32;
            a1[2] = v33;
            return a1;
          }
        }
      }
    }
  }
LABEL_2:
  if ( !a4 )
    goto LABEL_8;
  v9 = sub_A73ED0((_QWORD *)(a2 + 72), 23);
  v10 = (_QWORD *)(a2 + 72);
  if ( v9 || (v11 = sub_B49560(a2, 23), v10 = (_QWORD *)(a2 + 72), v11) )
  {
    if ( !(unsigned __int8)sub_A73ED0(v10, 4) && !(unsigned __int8)sub_B49560(a2, 4) )
      goto LABEL_8;
  }
  v12 = *(_QWORD *)(a2 - 32);
  if ( !v12
    || *(_BYTE *)v12
    || *(_QWORD *)(v12 + 24) != *(_QWORD *)(a2 + 80)
    || !sub_981210(*a4, v12, (unsigned int *)&v63)
    || (a4[((unsigned __int64)(unsigned int)v63 >> 6) + 1] & (1LL << v63)) != 0
    || (((int)*(unsigned __int8 *)(*a4 + ((unsigned int)v63 >> 2)) >> (2 * (v63 & 3))) & 3) == 0 )
  {
    goto LABEL_8;
  }
  if ( (_DWORD)v63 == 356 )
  {
LABEL_81:
    v55 = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
    if ( *(_BYTE *)v55 == 17 )
    {
      v26 = *(_QWORD *)(v55 + 24);
      if ( *(_DWORD *)(v55 + 32) > 0x40u )
LABEL_40:
        v26 = *(_QWORD *)v26;
LABEL_41:
      if ( v26 > 0x3FFFFFFFFFFFFFFBLL )
        v26 = 0xBFFFFFFFFFFFFFFELL;
      goto LABEL_43;
    }
LABEL_77:
    a1->m128i_i64[0] = v62;
    v26 = 0xBFFFFFFFFFFFFFFELL;
    goto LABEL_44;
  }
  if ( (unsigned int)v63 <= 0x164 )
  {
    if ( (_DWORD)v63 == 186 )
    {
LABEL_23:
      v21 = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
      if ( *(_BYTE *)v21 == 17 )
      {
        v22 = *(_QWORD *)(v21 + 24);
        if ( *(_DWORD *)(v21 + 32) > 0x40u )
          v22 = *(_QWORD *)v22;
        a1->m128i_i64[0] = v62;
        if ( v22 > 0x3FFFFFFFFFFFFFFBLL )
          v22 = 0xBFFFFFFFFFFFFFFELL;
        goto LABEL_28;
      }
      goto LABEL_98;
    }
    if ( (unsigned int)v63 <= 0xBA )
    {
      if ( (_DWORD)v63 != 121 && (_DWORD)v63 != 124 )
        goto LABEL_8;
      v56 = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
      v22 = 0xBFFFFFFFFFFFFFFELL;
      if ( *(_BYTE *)v56 != 17 )
      {
LABEL_56:
        a1->m128i_i64[0] = v62;
LABEL_28:
        v23 = _mm_loadu_si128(&v65);
        v24 = _mm_loadu_si128(v66);
        a1->m128i_i64[1] = v22;
        a1[1] = v23;
        a1[2] = v24;
        return a1;
      }
    }
    else
    {
      if ( (_DWORD)v63 != 355 )
        goto LABEL_8;
      v56 = *(_QWORD *)(a2 + 32 * (3LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
      if ( *(_BYTE *)v56 != 17 )
      {
LABEL_98:
        a1->m128i_i64[0] = v62;
        v22 = 0xBFFFFFFFFFFFFFFELL;
        goto LABEL_28;
      }
    }
    v22 = *(_QWORD *)(v56 + 24);
    if ( *(_DWORD *)(v56 + 32) > 0x40u )
      v22 = *(_QWORD *)v22;
    if ( v22 )
    {
      v57 = v22 | 0x8000000000000000LL;
      v58 = v22 <= 0x3FFFFFFFFFFFFFFBLL;
      v22 = 0xBFFFFFFFFFFFFFFELL;
      if ( v58 )
        v22 = v57;
    }
    goto LABEL_56;
  }
  if ( (_DWORD)v63 == 459 )
    goto LABEL_77;
  if ( (unsigned int)v63 <= 0x1CB )
  {
    if ( (_DWORD)v63 != 357 )
    {
      if ( (unsigned int)(v63 - 363) > 2 )
        goto LABEL_8;
      if ( a3 == 1 )
      {
        v60 = 4;
        if ( (_DWORD)v63 != 364 )
          v60 = 8 * ((_DWORD)v63 != 365) + 8;
        a1->m128i_i64[0] = v62;
        v26 = v60;
        goto LABEL_44;
      }
      goto LABEL_81;
    }
    goto LABEL_23;
  }
  if ( (_DWORD)v63 == 472 )
  {
    v25 = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
    v26 = 0xBFFFFFFFFFFFFFFELL;
    if ( *(_BYTE *)v25 != 17 )
    {
LABEL_43:
      a1->m128i_i64[0] = v62;
LABEL_44:
      v28 = _mm_loadu_si128(&v65);
      v29 = _mm_loadu_si128(v66);
      a1->m128i_i64[1] = v26;
      a1[1] = v28;
      a1[2] = v29;
      return a1;
    }
    v27 = *(_DWORD *)(v25 + 32);
    v26 = *(_QWORD *)(v25 + 24);
    if ( a3 )
    {
      if ( v27 > 0x40 )
        v26 = *(_QWORD *)v26;
      if ( v26 )
      {
        v61 = v26 | 0x8000000000000000LL;
        v58 = v26 <= 0x3FFFFFFFFFFFFFFBLL;
        v26 = 0xBFFFFFFFFFFFFFFELL;
        if ( v58 )
          v26 = v61;
      }
      goto LABEL_43;
    }
    if ( v27 > 0x40 )
      goto LABEL_40;
    goto LABEL_41;
  }
  if ( (unsigned int)v63 <= 0x1D8 && ((_DWORD)v63 == 463 || (_DWORD)v63 == 470) )
    goto LABEL_77;
LABEL_8:
  v13 = *(_DWORD *)(a2 + 4);
  v14 = _mm_loadu_si128(&v65);
  a1->m128i_i64[1] = -1;
  v15 = _mm_loadu_si128(v66);
  a1[1] = v14;
  a1[2] = v15;
  a1->m128i_i64[0] = *(_QWORD *)(a2 + 32 * (v6 - (v13 & 0x7FFFFFF)));
  return a1;
}
