// Function: sub_1D26C30
// Address: 0x1d26c30
//
__int64 __fastcall sub_1D26C30(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned int a4,
        unsigned int a5,
        unsigned int a6,
        unsigned int a7,
        unsigned int a8,
        unsigned __int8 a9,
        unsigned int a10,
        __int64 a11,
        __int64 a12)
{
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 (*v17)(); // rax
  char v18; // r14
  char i; // r13
  __int64 (*v20)(); // rax
  unsigned __int8 j; // r12
  __m128i *v22; // rcx
  char v23; // r8
  unsigned int v24; // r12d
  unsigned int v25; // r12d
  unsigned __int64 v26; // r11
  char v27; // di
  unsigned int v28; // eax
  bool v29; // zf
  __int64 v30; // rdx
  __int64 k; // r8
  __int64 (*v32)(); // rax
  char v33; // al
  unsigned int v34; // r10d
  unsigned int v35; // r10d
  __int64 v36; // r8
  __int64 (*v37)(); // rax
  __int64 (*v38)(); // rax
  __int64 (*v39)(); // rax
  char v40; // al
  char v42; // al
  unsigned int v43; // eax
  unsigned int v44; // eax
  unsigned int v45; // ecx
  __int32 v46; // eax
  __int64 v47; // rdx
  __int64 v48; // [rsp-10h] [rbp-B0h]
  __int64 v49; // [rsp-8h] [rbp-A8h]
  unsigned __int64 v50; // [rsp+10h] [rbp-90h]
  unsigned int v52; // [rsp+24h] [rbp-7Ch]
  unsigned int v54; // [rsp+30h] [rbp-70h]
  unsigned int v55; // [rsp+30h] [rbp-70h]
  unsigned __int8 v56; // [rsp+30h] [rbp-70h]
  int v57; // [rsp+34h] [rbp-6Ch]
  unsigned __int64 v58; // [rsp+38h] [rbp-68h]
  unsigned __int64 v59; // [rsp+38h] [rbp-68h]
  char v60; // [rsp+4Fh] [rbp-51h] BYREF
  __m128i v61; // [rsp+50h] [rbp-50h] BYREF
  __m128i v62; // [rsp+60h] [rbp-40h] BYREF

  v13 = a8;
  v14 = a5;
  v15 = a6;
  v52 = a2;
  v16 = a7;
  v17 = *(__int64 (**)())(*(_QWORD *)a12 + 456LL);
  if ( v17 != sub_1D12D70 )
  {
    a2 = a3;
    v46 = ((__int64 (__fastcall *)(__int64, unsigned __int64, _QWORD, __int64, _QWORD, _QWORD, _QWORD, __int64))v17)(
            a12,
            a3,
            a4,
            v14,
            (unsigned __int8)v15,
            (unsigned __int8)a7,
            (unsigned __int8)a8,
            a11);
    v61.m128i_i64[1] = v47;
    v13 = v48;
    v61.m128i_i32[0] = v46;
    v14 = v49;
    if ( (_BYTE)v46 != 1 )
      goto LABEL_14;
  }
  v61.m128i_i8[0] = 6;
  v18 = 6;
  v61.m128i_i64[1] = 0;
  if ( !a4 )
    goto LABEL_10;
  v58 = a3;
  for ( i = 6; ; v61.m128i_i8[0] = i )
  {
    if ( i )
    {
      if ( a4 >= (unsigned int)sub_1D13440(i) >> 3 )
        goto LABEL_9;
    }
    else if ( a4 >= (unsigned int)sub_1F58D40(&v61, a2, v13, v14, v15, v16) >> 3 )
    {
LABEL_9:
      v18 = i;
      a3 = v58;
      goto LABEL_10;
    }
    v20 = *(__int64 (**)())(*(_QWORD *)a12 + 448LL);
    if ( v20 != sub_1D12D60 )
      break;
LABEL_6:
    --i;
    v61.m128i_i64[1] = 0;
  }
  a2 = v61.m128i_u32[0];
  if ( !((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, _QWORD, _QWORD))v20)(
          a12,
          v61.m128i_u32[0],
          v61.m128i_i64[1],
          a10,
          a4,
          0) )
  {
    i = v61.m128i_i8[0];
    goto LABEL_6;
  }
  a3 = v58;
  v18 = v61.m128i_i8[0];
LABEL_10:
  for ( j = 6; !j || !*(_QWORD *)(a12 + 8LL * j + 120); --j )
    ;
  if ( v18 != j )
  {
    if ( v18 )
      sub_1D13440(v18);
    else
      sub_1F58D40(&v61, a2, v13, v14, v15, v16);
    v44 = sub_1D13440(j);
    if ( v44 < v45 )
    {
      v61.m128i_i8[0] = j;
      v61.m128i_i64[1] = 0;
    }
  }
LABEL_14:
  v57 = 0;
  v22 = &v62;
  if ( !a3 )
    return 1;
  while ( 2 )
  {
    if ( v61.m128i_i8[0] )
    {
      v24 = sub_1D13440(v61.m128i_i8[0]);
    }
    else
    {
      v43 = sub_1F58D40(&v61, a2, v13, v22, 0, v16);
      v23 = 0;
      v24 = v43;
    }
    v25 = v24 >> 3;
    v26 = v25;
    if ( a3 < v25 )
    {
      v27 = v23;
      v62 = _mm_loadu_si128(&v61);
      if ( v23 )
      {
LABEL_19:
        if ( (unsigned __int8)(v27 - 8) <= 0x65u )
        {
          v28 = sub_1D13440(v27);
          goto LABEL_21;
        }
LABEL_31:
        for ( k = (unsigned int)v62.m128i_u8[0] - 1; ; k = (unsigned int)v13 )
        {
          v62.m128i_i8[0] = k;
          v62.m128i_i64[1] = 0;
          if ( (_BYTE)k == 3 )
            break;
          v37 = *(__int64 (**)())(*(_QWORD *)a12 + 464LL);
          if ( v37 != sub_1D12D80 )
          {
            v56 = k;
            a2 = (unsigned int)k;
            v42 = ((__int64 (__fastcall *)(__int64, _QWORD))v37)(a12, (unsigned int)k);
            k = v56;
            v13 = (unsigned int)v56 - 1;
            if ( !v42 )
              continue;
          }
          if ( (_BYTE)k )
            goto LABEL_33;
          v34 = sub_1F58D40(&v62, a2, v13, v22, k, v16);
          goto LABEL_34;
        }
        goto LABEL_33;
      }
      while ( 1 )
      {
        if ( !(unsigned __int8)sub_1F58D20(&v61) && !(unsigned __int8)sub_1F58CD0(&v61) )
          goto LABEL_31;
        v28 = sub_1F58D40(&v61, a2, v13, v22, v36, v16);
LABEL_21:
        if ( v28 > 0x40 )
        {
          v29 = *(_QWORD *)(a12 + 168) == 0;
          v62.m128i_i8[0] = 6;
          v62.m128i_i64[1] = 0;
          if ( v29 || (*(_BYTE *)(a12 + 4162) & 0xFB) != 0 )
            goto LABEL_44;
          a2 = 6;
          v30 = 6;
        }
        else
        {
          v29 = *(_QWORD *)(a12 + 160) == 0;
          v62.m128i_i8[0] = 5;
          v62.m128i_i64[1] = 0;
          if ( v29 || (*(_BYTE *)(a12 + 3903) & 0xFB) != 0 )
            goto LABEL_31;
          a2 = 5;
          v30 = 5;
        }
        LOBYTE(k) = v62.m128i_i8[0];
        v32 = *(__int64 (**)())(*(_QWORD *)a12 + 464LL);
        if ( v32 == sub_1D12D80 )
          goto LABEL_33;
        v54 = v30;
        v33 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __m128i *, _QWORD))v32)(
                a12,
                a2,
                v30,
                v22,
                v62.m128i_u8[0]);
        LOBYTE(k) = v62.m128i_i8[0];
        v13 = v54;
        if ( v33 )
          goto LABEL_33;
        if ( v54 != 6 )
          goto LABEL_31;
LABEL_44:
        if ( !*(_QWORD *)(a12 + 200) )
          goto LABEL_31;
        if ( (*(_BYTE *)(a12 + 5198) & 0xFB) != 0 )
          goto LABEL_31;
        v38 = *(__int64 (**)())(*(_QWORD *)a12 + 464LL);
        if ( v38 != sub_1D12D80 )
        {
          a2 = 10;
          if ( !((unsigned __int8 (__fastcall *)(__int64, __int64))v38)(a12, 10) )
            goto LABEL_31;
        }
        v62.m128i_i8[0] = 10;
        LOBYTE(k) = 10;
LABEL_33:
        v34 = sub_1D13440(k);
LABEL_34:
        v35 = v34 >> 3;
        v26 = v35;
        if ( v25 > 7 && (a9 & (v57 != 0)) != 0 && a3 > v35 )
        {
          v39 = *(__int64 (**)())(*(_QWORD *)a12 + 448LL);
          if ( v39 != sub_1D12D60 )
          {
            v50 = v35;
            v55 = v35;
            a2 = v61.m128i_u32[0];
            v40 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, _QWORD, char *))v39)(
                    a12,
                    v61.m128i_u32[0],
                    v61.m128i_i64[1],
                    a10,
                    a4,
                    &v60);
            v35 = v55;
            v26 = v50;
            if ( v40 )
            {
              if ( v60 )
              {
                v26 = a3;
                break;
              }
            }
          }
        }
        v61 = _mm_loadu_si128(&v62);
        if ( v26 <= a3 )
          break;
        v27 = v61.m128i_i8[0];
        v25 = v35;
        v62 = _mm_loadu_si128(&v61);
        if ( v61.m128i_i8[0] )
          goto LABEL_19;
      }
    }
    if ( ++v57 <= v52 )
    {
      a2 = *(_QWORD *)(a1 + 8);
      if ( a2 == *(_QWORD *)(a1 + 16) )
      {
        v59 = v26;
        sub_1D26AB0((const __m128i **)a1, (const __m128i *)a2, &v61);
        v26 = v59;
      }
      else
      {
        if ( a2 )
        {
          *(__m128i *)a2 = _mm_loadu_si128(&v61);
          a2 = *(_QWORD *)(a1 + 8);
        }
        a2 += 16;
        *(_QWORD *)(a1 + 8) = a2;
      }
      a3 -= v26;
      if ( !a3 )
        return 1;
      continue;
    }
    return 0;
  }
}
