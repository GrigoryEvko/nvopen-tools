// Function: sub_7F9500
// Address: 0x7f9500
//
__m128i *__fastcall sub_7F9500(__int64 a1, __int64 a2, int a3)
{
  __int64 v5; // r12
  _BYTE *v6; // r12
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // rdi
  __m128i *v11; // rdi
  _BYTE *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rdi
  _QWORD *v20; // r14
  _QWORD *v21; // r12
  _QWORD *v22; // rdi
  __int64 v23; // rsi
  _BYTE *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rax
  __int64 v30; // rax
  char i; // dl
  _QWORD v32[5]; // [rsp+0h] [rbp-B0h] BYREF
  char v33; // [rsp+28h] [rbp-88h]
  _BYTE v34[32]; // [rsp+30h] [rbp-80h] BYREF
  _QWORD *v35; // [rsp+50h] [rbp-60h]

  v5 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    v6 = *(_BYTE **)(v5 + 32);
    if ( v6 )
      return (__m128i *)v6;
    v18 = *(_QWORD *)(qword_4F04C50 + 32LL);
    v19 = *(_QWORD *)(*(_QWORD *)(qword_4F04C50 + 40LL) + 112LL);
    if ( (((*(_BYTE *)(v18 + 205) & 0x1C) - 8) & 0xF4) == 0
      && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v18 + 40) + 32LL) + 176LL) & 0x10) != 0 )
    {
      v19 = *(_QWORD *)(v19 + 112);
    }
    sub_7F90D0(v19, (__int64)v34);
    sub_7F55E0(*(_QWORD *)a1, (__int64)v34, (__int64)v32);
    return sub_7F9430((__int64)v34, a3, 0);
  }
  v8 = *(_QWORD *)(a1 + 8);
  if ( v8 )
  {
    v9 = *(_QWORD *)(v8 + 16);
    if ( (*(_BYTE *)(v8 + 32) & 2) == 0 )
      v5 = *(_QWORD *)(v8 + 8);
    if ( v9 )
    {
      sub_7F90D0(*(_QWORD *)(qword_4F04C50 + 64LL), (__int64)v34);
      v10 = *(_QWORD *)(v9 + 120);
      v32[2] = 0;
      v32[4] = 0;
      v32[0] = v35;
      v33 = 0;
      v35 = v32;
      v32[3] = v9;
      v32[1] = v10;
      if ( (unsigned int)sub_8D2FB0(v10)
        || v5
        && (*(_BYTE *)(v5 + 172) & 1) != 0
        && (*(_BYTE *)(*(_QWORD *)(a1 + 8) + 32LL) & 8) == 0
        && (unsigned int)sub_8D2E30(*(_QWORD *)(v9 + 120)) )
      {
        v11 = sub_7F9430((__int64)v34, 0, 0);
        v12 = sub_73DCD0(v11);
        v6 = v12;
        if ( !a3 )
          return (__m128i *)sub_731370((__int64)v12, 0, v13, v14, v15, v16);
        return (__m128i *)v6;
      }
      return sub_7F9430((__int64)v34, a3, 0);
    }
    if ( (unsigned int)sub_8D2FB0(*(_QWORD *)(v5 + 120))
      || (*(_BYTE *)(v5 + 172) & 1) != 0 && (*(_BYTE *)(*(_QWORD *)(a1 + 8) + 32LL) & 8) == 0
      || *(char *)(v5 + 169) < 0 && (v29 = *(_QWORD *)(v5 + 128)) != 0 && (*(_BYTE *)(v29 + 32) & 1) != 0 )
    {
      v30 = *(_QWORD *)(v5 + 120);
      for ( i = *(_BYTE *)(v30 + 140); i == 12; i = *(_BYTE *)(v30 + 140) )
        v30 = *(_QWORD *)(v30 + 160);
      v17 = v5;
      if ( i == 6 )
      {
        sub_7F90D0(v5, (__int64)v34);
        return sub_7F9430((__int64)v34, a3, 0);
      }
    }
    else
    {
      if ( qword_4F04C50 && *(_QWORD *)(qword_4F04C50 + 72LL) == v5 )
      {
        sub_7F90D0(qword_4D03F58, (__int64)v34);
        return sub_7F9430((__int64)v34, a3, 0);
      }
      v17 = v5;
    }
    sub_7F9080(v17, (__int64)v34);
    return sub_7F9430((__int64)v34, a3, 0);
  }
  if ( !*(_DWORD *)(a1 + 16) )
    sub_721090();
  v20 = *(_QWORD **)(*(_QWORD *)(a2 + 8) + 120LL);
  v21 = sub_7DDF30(a1, a2);
  v22 = v20;
  if ( (unsigned int)sub_8D2FB0(v20) )
    v22 = (_QWORD *)sub_8D46C0(v20);
  v23 = sub_72D2E0(v22);
  v6 = sub_73E130(v21, v23);
  if ( (unsigned int)sub_8D2FB0(v20) )
    return (__m128i *)v6;
  v24 = sub_73DCD0(v6);
  v6 = v24;
  if ( a3 )
    return (__m128i *)v6;
  return (__m128i *)sub_731370((__int64)v24, v23, v25, v26, v27, v28);
}
