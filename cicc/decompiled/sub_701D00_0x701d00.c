// Function: sub_701D00
// Address: 0x701d00
//
void __fastcall sub_701D00(
        _QWORD *a1,
        __int64 a2,
        char a3,
        int a4,
        int a5,
        char a6,
        char a7,
        char a8,
        char a9,
        int a10,
        int a11,
        __int64 *a12,
        _DWORD *a13,
        _QWORD *a14,
        __m128i *a15,
        int *a16,
        __int64 *a17)
{
  __int64 v18; // rax
  __int64 v20; // rsi
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 *v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdi
  __int64 v27; // r9
  __int64 v28; // rax
  char i; // dl
  __int64 v30; // r15
  bool v31; // bl
  int v32; // ebx
  __int64 v33; // rdi
  int v34; // ebx
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  int v39; // ebx
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 v44; // rdi
  __int64 v45; // r8
  __int64 v46; // rcx
  __int64 v47; // rax
  char v48; // al
  char v49; // al
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  _BOOL4 v54; // eax
  _BOOL4 v55; // edx
  __int64 v59; // [rsp+28h] [rbp-58h] BYREF
  __m128i v60; // [rsp+30h] [rbp-50h] BYREF
  __m128i v61[4]; // [rsp+40h] [rbp-40h] BYREF

  v18 = sub_72B0F0(a1, &v59);
  v60.m128i_i64[0] = v18;
  if ( HIDWORD(qword_4F077B4) )
  {
    if ( v59 )
    {
      v20 = *(_QWORD *)(v18 + 256);
      if ( v20 )
      {
        v21 = *(_QWORD *)(v20 + 24);
        if ( v21 )
        {
          if ( (*(_BYTE *)(v18 + 203) & 0x20) == 0 )
            *(_QWORD *)(v59 + 56) = v21;
        }
      }
    }
  }
  v60.m128i_i64[1] = 0;
  v22 = sub_6FD870(a1, a2, v18, a3, a4, a5, a6, a7, a8, a9, a11, a13, &v60.m128i_i64[1]);
  v23 = (__int64 *)v22;
  if ( !a13 )
  {
    *(_QWORD *)(v22 + 28) = *a12;
    *(_QWORD *)(v22 + 36) = *a12;
    *(_QWORD *)(v22 + 44) = *a14;
    v24 = v60.m128i_i64[1];
    if ( v60.m128i_i64[1] )
      goto LABEL_25;
LABEL_26:
    v24 = 0;
    if ( a17 )
      goto LABEL_14;
LABEL_27:
    sub_6E70E0(v23, (__int64)a15);
    v33 = a15->m128i_i64[0];
    *(__int64 *)((char *)a15[4].m128i_i64 + 4) = *(_QWORD *)a13;
    if ( (unsigned int)sub_8D32E0(v33) )
    {
      v34 = sub_8D3110(a15->m128i_i64[0]);
      sub_6F82C0((__int64)a15, (__int64)a15, v35, v36, v37, v38);
      if ( v34 )
        sub_6ED1A0((__int64)a15);
    }
    goto LABEL_19;
  }
  v24 = v60.m128i_i64[1];
  if ( *a13 )
    v25 = *(_QWORD *)a13;
  else
    v25 = *a12;
  *(__int64 *)((char *)v23 + 28) = v25;
  *(__int64 *)((char *)v23 + 36) = *a12;
  *(__int64 *)((char *)v23 + 44) = *a14;
  if ( !v24 )
    goto LABEL_26;
  if ( *a13 )
  {
    *(_QWORD *)(v24 + 28) = *(_QWORD *)a13;
    goto LABEL_13;
  }
LABEL_25:
  *(_QWORD *)(v24 + 28) = *a12;
LABEL_13:
  *(_QWORD *)(v24 + 36) = *a12;
  *(_QWORD *)(v24 + 44) = *a14;
  if ( !a17 )
    goto LABEL_27;
LABEL_14:
  *a17 = v24;
  sub_6E70E0(v23, (__int64)a15);
  v26 = a15->m128i_i64[0];
  *(__int64 *)((char *)a15[4].m128i_i64 + 4) = *(_QWORD *)a13;
  if ( (unsigned int)sub_8D32E0(v26) )
  {
    v39 = sub_8D3110(a15->m128i_i64[0]);
    sub_6F82C0((__int64)a15, (__int64)a15, v40, v41, v42, v43);
    if ( v39 )
      sub_6ED1A0((__int64)a15);
  }
  if ( !a15[1].m128i_i8[0] )
    goto LABEL_19;
  v28 = a15->m128i_i64[0];
  for ( i = *(_BYTE *)(a15->m128i_i64[0] + 140); i == 12; i = *(_BYTE *)(v28 + 140) )
    v28 = *(_QWORD *)(v28 + 160);
  v30 = *a17;
  v31 = i != 0 && *a17 != 0;
  if ( !v31 )
  {
LABEL_19:
    v32 = 0;
    goto LABEL_20;
  }
  v44 = v60.m128i_i64[0];
  v61[0] = 0u;
  v45 = v60.m128i_i64[0];
  if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) == 0 )
  {
LABEL_63:
    if ( !v44 )
      goto LABEL_74;
    v48 = *(_BYTE *)(v44 + 202) & 4;
    goto LABEL_65;
  }
  if ( !v60.m128i_i64[0] )
  {
    if ( !word_4D04898 )
      goto LABEL_74;
    goto LABEL_59;
  }
  if ( !*(_BYTE *)(v60.m128i_i64[0] + 174) && *(_WORD *)(v60.m128i_i64[0] + 176) )
  {
    v46 = *(_QWORD *)(v60.m128i_i64[0] + 152);
    if ( *(_BYTE *)(v46 + 140) == 12 )
    {
      v47 = *(_QWORD *)(v60.m128i_i64[0] + 152);
      do
        v47 = *(_QWORD *)(v47 + 160);
      while ( *(_BYTE *)(v47 + 140) == 12 );
      if ( (*(_BYTE *)(*(_QWORD *)(v47 + 168) + 16LL) & 1) == 0 )
      {
        do
          v46 = *(_QWORD *)(v46 + 160);
        while ( *(_BYTE *)(v46 + 140) == 12 );
LABEL_41:
        if ( (*(_BYTE *)(*(_QWORD *)(v46 + 168) + 16LL) & 2) == 0 )
          goto LABEL_42;
LABEL_59:
        if ( *(char *)(qword_4D03C50 + 19LL) >= 0 || dword_4D048AC )
        {
          if ( (unsigned int)sub_7016D0(v30, &v60, a15, v61, v60.m128i_i64[0], v27) )
            goto LABEL_57;
          v44 = v60.m128i_i64[0];
          v45 = v60.m128i_i64[0];
        }
        goto LABEL_63;
      }
    }
    else if ( (*(_BYTE *)(*(_QWORD *)(v46 + 168) + 16LL) & 1) == 0 )
    {
      goto LABEL_41;
    }
  }
LABEL_42:
  v48 = *(_BYTE *)(v60.m128i_i64[0] + 202) & 4;
  if ( word_4D04898 )
  {
    if ( !v48 )
    {
      v49 = *(_BYTE *)(v60.m128i_i64[0] + 193);
      if ( (v49 & 6) == 0 && ((*(_BYTE *)(v60.m128i_i64[0] + 192) & 2) == 0 || !dword_4D04888 || a4) )
        goto LABEL_50;
      if ( (v49 & 4) != 0 && (*(_BYTE *)(qword_4D03C50 + 20LL) & 2) != 0 )
        goto LABEL_50;
    }
    goto LABEL_59;
  }
LABEL_65:
  if ( v48 && dword_4F077C0 && (qword_4F077A8 <= 0x9E33u || (*(_BYTE *)(qword_4D03C50 + 19LL) & 4) != 0) )
    v45 = *(_QWORD *)(*(_QWORD *)(v44 + 256) + 8LL);
LABEL_50:
  if ( *(_BYTE *)(v45 + 174) || !*(_WORD *)(v45 + 176) )
  {
LABEL_71:
    if ( (*(_BYTE *)(v44 + 193) & 4) != 0 )
    {
      if ( dword_4D0488C )
      {
LABEL_73:
        v54 = sub_6E5AC0();
        v44 = v60.m128i_i64[0];
        v55 = v54;
LABEL_82:
        v32 = 0;
        sub_6F50A0(v44, a15, v55, v61, 0, v27);
        goto LABEL_83;
      }
LABEL_76:
      if ( !word_4D04898 )
        goto LABEL_81;
      v27 = (unsigned int)qword_4F077B4;
      if ( !(_DWORD)qword_4F077B4 || qword_4F077A0 <= 0x765Bu )
        goto LABEL_81;
      if ( !(unsigned int)sub_729F80(dword_4F063F8) )
      {
        v44 = v60.m128i_i64[0];
LABEL_81:
        v55 = 0;
        goto LABEL_82;
      }
      if ( v31 )
        goto LABEL_73;
      v44 = v60.m128i_i64[0];
LABEL_75:
      v55 = 1;
      goto LABEL_82;
    }
LABEL_74:
    v31 = 0;
    if ( dword_4D0488C )
      goto LABEL_75;
    goto LABEL_76;
  }
  if ( !(unsigned int)sub_7176C0(v45, 0) )
    goto LABEL_70;
  if ( (dword_4F04C44 != -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0)
    && (unsigned int)sub_696840((__int64)a15) )
  {
    sub_6F4B70(a15, 0, v50, v51, v52, v53);
    goto LABEL_57;
  }
  if ( !(unsigned int)sub_690CC0(a15, v30) )
  {
LABEL_70:
    v44 = v60.m128i_i64[0];
    if ( !v60.m128i_i64[0] )
      goto LABEL_74;
    goto LABEL_71;
  }
LABEL_57:
  v32 = 1;
LABEL_83:
  sub_67E3D0(v61);
LABEL_20:
  if ( a16 )
    *a16 = v32;
}
