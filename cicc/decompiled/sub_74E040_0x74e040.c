// Function: sub_74E040
// Address: 0x74e040
//
__int64 __fastcall sub_74E040(__int64 a1, int a2, int a3, __int64 a4)
{
  const __m128i *v6; // r13
  __int8 v7; // al
  const __m128i *i; // r15
  char v9; // dl
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // r13
  __int64 j; // rsi
  __int64 k; // r14
  __int64 v18; // rcx
  __int64 v19; // r8
  int v20; // eax
  __int64 m; // rdx
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 v24; // rt2
  __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // r8
  int v28; // eax
  __int64 result; // rax
  void (__fastcall *v30)(char *, __int64); // rax
  __int64 v31; // rdi
  unsigned int v32; // eax
  void (__fastcall *v33)(char *, __int64); // rax
  bool v34; // zf
  int v35; // eax
  int v36; // eax
  int v37; // [rsp+4h] [rbp-14Ch]
  int v38; // [rsp+4h] [rbp-14Ch]
  int v39; // [rsp+20h] [rbp-130h]
  unsigned __int64 v40; // [rsp+20h] [rbp-130h]
  unsigned int v41; // [rsp+28h] [rbp-128h]
  _BOOL4 v42; // [rsp+2Ch] [rbp-124h]
  int v45; // [rsp+38h] [rbp-118h]
  int v46; // [rsp+48h] [rbp-108h] BYREF
  int v47; // [rsp+4Ch] [rbp-104h] BYREF
  __int64 v48; // [rsp+50h] [rbp-100h] BYREF
  __int64 v49; // [rsp+58h] [rbp-F8h] BYREF
  _OWORD v50[10]; // [rsp+60h] [rbp-F0h] BYREF
  __m128i v51; // [rsp+100h] [rbp-50h]
  __m128i v52; // [rsp+110h] [rbp-40h]

  v6 = *(const __m128i **)(a1 + 128);
  v7 = v6[8].m128i_i8[12];
  for ( i = v6; v7 == 12; v7 = i[8].m128i_i8[12] )
    i = (const __m128i *)i[10].m128i_i64[0];
  v9 = *(_BYTE *)(a1 + 168);
  if ( (v9 & 8) != 0 && v7 != 6 )
  {
    v37 = 1;
    v11 = 0;
    goto LABEL_8;
  }
  if ( (!*(_BYTE *)(a4 + 136) || !*(_BYTE *)(a4 + 141)) && (*(_DWORD *)(a1 + 168) & 0x1000028) == 8 )
  {
    v37 = 0;
    v11 = 0;
    if ( *(_BYTE *)(a1 + 176) != 1 )
      goto LABEL_8;
    for ( j = *(_QWORD *)(*(_QWORD *)(a1 + 184) + 120LL); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    for ( k = sub_8D46C0(i); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
      ;
    v37 = sub_8D3410(j);
    if ( !v37 && (!(unsigned int)sub_8D3A70(j) || j == k || (unsigned int)sub_8D97D0(k, j, 0, v18, v19)) )
    {
      v9 = *(_BYTE *)(a1 + 168);
      v11 = 0;
      goto LABEL_8;
    }
  }
  v10 = sub_8D46C0(i);
  v37 = 0;
  v9 = *(_BYTE *)(a1 + 168);
  v11 = v10;
LABEL_8:
  v42 = 0;
  if ( (v9 & 0x40) != 0 )
  {
    v42 = 1;
    if ( *(_BYTE *)(a4 + 136) )
      v42 = *(_BYTE *)(a4 + 141) == 0;
  }
  sub_74C8A0(a1, v11, a2 ^ 1, 0, 0, &v48, &v46, &v49, &v47, a4);
  v45 = v46;
  if ( v46 )
    goto LABEL_12;
  v20 = sub_8D3410(v48);
  if ( (((unsigned __int8)a2 ^ 1) & 1) != 0 && v20 )
  {
    v14 = v48;
    if ( !*(_BYTE *)(a4 + 137) )
    {
      if ( *(_BYTE *)(a4 + 136) && *(_BYTE *)(a1 + 176) == 2 && *(_BYTE *)(*(_QWORD *)(a1 + 184) + 173LL) == 2 )
      {
        v34 = *(_BYTE *)(a4 + 157) == 0;
        v46 = v34;
        if ( !v34 )
        {
          v39 = v37;
LABEL_110:
          v13 = v49;
          if ( !v49 )
            goto LABEL_45;
          goto LABEL_39;
        }
LABEL_109:
        v39 = 1;
        v48 = sub_8D4050(v48);
        v14 = v48;
        goto LABEL_110;
      }
      if ( !v49 )
      {
        if ( !v46 )
        {
          v45 = 0;
          v39 = v37;
          goto LABEL_45;
        }
        goto LABEL_109;
      }
    }
    v46 = 1;
    goto LABEL_109;
  }
  if ( !v46 )
    goto LABEL_36;
LABEL_12:
  if ( !(unsigned int)sub_8D2E30(v6) )
  {
LABEL_36:
    v13 = v49;
    v14 = v48;
    if ( !v49 )
    {
      v45 = 0;
      v39 = v37;
      goto LABEL_45;
    }
    v45 = a2;
    v39 = a2;
    if ( a2 )
      goto LABEL_45;
    goto LABEL_38;
  }
  if ( v11 )
  {
    if ( (unsigned int)sub_8D2310(v11)
      && (dword_4F077C4 != 2 || !(unsigned int)sub_8D9600(v11, sub_745090, 4615))
      && !*(_BYTE *)(a1 + 176) )
    {
      v12 = *(_QWORD *)(a1 + 184);
      if ( (*(_BYTE *)(v12 + 89) & 4) != 0 && dword_4F077C4 == 2 )
        sub_8D9600(*(_QWORD *)(*(_QWORD *)(v12 + 40) + 32LL), sub_745090, 4615);
    }
    v13 = v49;
    v14 = v48;
    if ( !v49 )
    {
      v45 = 0;
      v15 = sub_8D21C0(v48);
      v39 = v37;
      goto LABEL_46;
    }
    if ( a2 )
    {
      v15 = sub_8D21C0(v48);
      v45 = a2;
      v39 = a2;
      goto LABEL_46;
    }
  }
  else
  {
    v13 = v49;
    v14 = v48;
    if ( !v49 )
    {
      v45 = 0;
      v15 = sub_8D21C0(v48);
      v39 = v37;
      goto LABEL_48;
    }
    if ( a2 )
    {
      v15 = sub_8D21C0(v48);
      if ( *(_BYTE *)(a1 + 173) != 6 || *(_BYTE *)(a1 + 176) != 1 )
      {
        v45 = a2;
        goto LABEL_61;
      }
      v39 = a2;
      v45 = a2;
LABEL_85:
      if ( (unsigned int)sub_8D3410(v15) && (*(_WORD *)(v15 + 168) & 0x180) == 0 && *(_QWORD *)(v15 + 176) )
      {
        v31 = sub_8D40F0(v15);
        if ( *(_BYTE *)(v31 + 140) == 12 && (sub_8D4C10(v31, 1) & 1) != 0 )
          goto LABEL_61;
        if ( unk_4F068C0 && (*(_BYTE *)(*(_QWORD *)(a1 + 184) + 170LL) & 0x10) != 0 && !*(_QWORD *)(a1 + 192) )
          goto LABEL_61;
      }
      goto LABEL_49;
    }
  }
LABEL_38:
  v39 = v37;
LABEL_39:
  for ( m = v14; *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
    ;
  v22 = *(_QWORD *)(m + 128);
  if ( !v22 || (v24 = v13 % v22, v23 = v13 / v22, v24) )
  {
    v45 = 1;
    v39 = 1;
  }
  else
  {
    v49 = v23;
    v45 = 0;
  }
LABEL_45:
  v15 = sub_8D21C0(v14);
  if ( !v11 )
    goto LABEL_48;
LABEL_46:
  v25 = sub_8D21C0(v11);
  if ( v25 == v15 || (unsigned int)sub_8D97D0(v15, v25, 0, v26, v27) )
  {
LABEL_48:
    if ( *(_BYTE *)(a1 + 173) != 6 || *(_BYTE *)(a1 + 176) != 1 )
      goto LABEL_49;
    goto LABEL_85;
  }
  if ( (*(_BYTE *)(a1 + 168) & 8) != 0 || *(_BYTE *)(a1 + 176) )
  {
    if ( dword_4F077C4 != 2 && (unsigned int)sub_8E3210(v11) )
    {
      v41 = 1;
      if ( a3 )
      {
        (*(void (__fastcall **)(char *, __int64))a4)("(", a4);
        v41 = a3;
        a3 = 1;
      }
      (*(void (__fastcall **)(const char *, __int64))a4)("(void *)", a4);
      v42 = 0;
LABEL_66:
      if ( !v49 )
      {
        if ( !v45 )
        {
          v38 = 0;
          goto LABEL_69;
        }
        v36 = a3;
        a3 = 0;
        v39 = v36;
        v45 = v41;
        goto LABEL_53;
      }
      goto LABEL_94;
    }
    if ( !*(_BYTE *)(a4 + 156) || !*(_BYTE *)(a4 + 136) )
    {
LABEL_61:
      v41 = 1;
      if ( a3 )
      {
        (*(void (__fastcall **)(char *, __int64))a4)("(", a4);
        v41 = a3;
        a3 = 1;
      }
      if ( a2 )
      {
        v50[0] = _mm_loadu_si128(i);
        v50[1] = _mm_loadu_si128(i + 1);
        v50[2] = _mm_loadu_si128(i + 2);
        v50[3] = _mm_loadu_si128(i + 3);
        v50[4] = _mm_loadu_si128(i + 4);
        v50[5] = _mm_loadu_si128(i + 5);
        v50[6] = _mm_loadu_si128(i + 6);
        v50[7] = _mm_loadu_si128(i + 7);
        v50[8] = _mm_loadu_si128(i + 8);
        v50[9] = _mm_loadu_si128(i + 9);
        v51 = _mm_loadu_si128(i + 10);
        v52 = _mm_loadu_si128(i + 11);
        if ( v49 || dword_4F072C8 )
        {
          (*(void (__fastcall **)(char *, __int64))a4)("*", a4);
          v51.m128i_i8[8] &= ~1u;
          (*(void (__fastcall **)(char *, __int64))a4)("(", a4);
          sub_74B930((__int64)v50, a4);
          (*(void (__fastcall **)(char *, __int64))a4)(")", a4);
          v46 = 0;
          v42 = 0;
          a2 = 0;
        }
        else
        {
          v51.m128i_i8[8] |= 1u;
          sub_74DB20((__int64)v50, v42, (_QWORD *)a4);
        }
      }
      else
      {
        sub_74DB20((__int64)i, v42, (_QWORD *)a4);
        if ( v37 )
        {
          if ( !(unsigned int)sub_8D2930(i) || (v40 = i[8].m128i_u64[0], v40 < sub_88CEE0(v48, v50)) )
            (*(void (__fastcall **)(char *, __int64))a4)("(unsigned long)", a4);
        }
      }
      goto LABEL_66;
    }
  }
LABEL_49:
  if ( v39 )
    goto LABEL_61;
  if ( v49 )
  {
    if ( !a3 )
    {
      v42 = 0;
      if ( !v45 )
      {
        v38 = 0;
        v41 = 1;
        goto LABEL_69;
      }
      goto LABEL_53;
    }
    v32 = a3;
    a3 = 0;
    v42 = 0;
    v41 = v32;
LABEL_94:
    (*(void (__fastcall **)(char *, __int64))a4)("(", a4);
    if ( !v45 )
    {
      v38 = 1;
      goto LABEL_69;
    }
    v39 = a3;
    a3 = v45;
    v45 = v41;
    goto LABEL_53;
  }
  if ( !v45 )
  {
    v42 = 0;
    v38 = 0;
    v41 = a3;
    a3 = 0;
    goto LABEL_69;
  }
  v42 = 0;
  if ( !a3 )
  {
    v38 = 0;
    goto LABEL_54;
  }
  v35 = a3;
  a3 = 0;
  v45 = v35;
LABEL_53:
  (*(void (__fastcall **)(char *, __int64))a4)("(", a4);
  v28 = a3;
  a3 = 1;
  v38 = v28;
LABEL_54:
  (*(void (__fastcall **)(char *, __int64))a4)("(char *)", a4);
  v41 = v45;
  v45 = a3;
  a3 = v39;
LABEL_69:
  if ( a2 || *(_BYTE *)(a4 + 157) )
    goto LABEL_71;
  if ( (unsigned int)sub_8D2FB0(i) && !*(_BYTE *)(a4 + 136) )
  {
    if ( (unsigned int)sub_8D3110(i) )
      (*(void (__fastcall **)(const char *, __int64))a4)("rvalue reference to ", a4);
    else
      (*(void (__fastcall **)(char *, __int64))a4)("reference to ", a4);
    goto LABEL_71;
  }
  if ( v46 )
  {
LABEL_71:
    result = (__int64)sub_74C8A0(a1, v11, a2 ^ 1u, v47 == 0, 1, &v48, &v46, (__int64 *)v50, &v47, a4);
    goto LABEL_72;
  }
  if ( v41 )
  {
    (*(void (__fastcall **)(char *, __int64))a4)("(", a4);
    v41 = 1;
  }
  v33 = *(void (__fastcall **)(char *, __int64))a4;
  if ( *(_BYTE *)(a1 + 173) == 6 && *(_BYTE *)(a1 + 176) == 6 )
    v33("&&", a4);
  else
    v33("&", a4);
  sub_74C8A0(a1, v11, 1u, v47 == 0, 1, &v48, &v46, (__int64 *)v50, &v47, a4);
  result = v41;
  if ( v41 )
    result = (*(__int64 (__fastcall **)(char *, __int64))a4)(")", a4);
LABEL_72:
  if ( v45 )
    result = (*(__int64 (__fastcall **)(char *, __int64))a4)(")", a4);
  if ( v49 )
  {
    v30 = *(void (__fastcall **)(char *, __int64))a4;
    if ( v49 < 0 )
      v30(" ", a4);
    else
      v30(" + ", a4);
    result = sub_7451C0(v49, (__int64 (__fastcall **)(char *, _QWORD))a4);
    if ( v38 )
      result = (*(__int64 (__fastcall **)(char *, __int64))a4)(")", a4);
  }
  if ( v42 )
    result = (*(__int64 (__fastcall **)(char *, __int64))a4)(")", a4);
  if ( a3 )
    return (*(__int64 (__fastcall **)(char *, __int64))a4)(")", a4);
  return result;
}
