// Function: sub_F094D0
// Address: 0xf094d0
//
__int64 __fastcall sub_F094D0(__int64 a1, _BYTE *a2)
{
  __int64 v2; // r15
  __int64 v4; // r14
  __int64 v5; // r12
  unsigned __int64 v6; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  __m128i v11; // xmm2
  __m128i v12; // xmm3
  _BYTE *v13; // rax
  unsigned __int8 *v14; // rdi
  unsigned __int64 v15; // rcx
  unsigned __int8 *v16; // rax
  _BYTE *v17; // rbx
  unsigned __int8 *v18; // rdx
  int v19; // eax
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rdi
  __int64 v23; // rbx
  unsigned __int8 *v24; // rdi
  __int64 *v25; // rax
  __int64 v26; // r14
  __int64 *v27; // rax
  __int64 *v28; // rax
  __int64 v29; // [rsp+8h] [rbp-168h]
  __int64 v30; // [rsp+10h] [rbp-160h]
  unsigned __int8 *v31; // [rsp+18h] [rbp-158h]
  __int64 v32; // [rsp+20h] [rbp-150h]
  __int64 v33; // [rsp+20h] [rbp-150h]
  unsigned __int64 v34; // [rsp+20h] [rbp-150h]
  __int64 v35; // [rsp+20h] [rbp-150h]
  unsigned __int8 *v36; // [rsp+28h] [rbp-148h]
  __int64 v37; // [rsp+30h] [rbp-140h]
  unsigned __int64 v38; // [rsp+40h] [rbp-130h]
  unsigned __int64 v39; // [rsp+50h] [rbp-120h]
  _BYTE v40[16]; // [rsp+60h] [rbp-110h] BYREF
  void (__fastcall *v41)(_BYTE *, _BYTE *, __int64); // [rsp+70h] [rbp-100h]
  unsigned __int8 (__fastcall *v42)(_BYTE *, _BYTE *); // [rsp+78h] [rbp-F8h]
  _OWORD v43[2]; // [rsp+80h] [rbp-F0h] BYREF
  _BYTE v44[16]; // [rsp+A0h] [rbp-D0h] BYREF
  void (__fastcall *v45)(_BYTE *, _BYTE *, __int64); // [rsp+B0h] [rbp-C0h]
  __int64 v46; // [rsp+B8h] [rbp-B8h]
  __m128i v47; // [rsp+C0h] [rbp-B0h] BYREF
  __m128i v48; // [rsp+D0h] [rbp-A0h] BYREF
  _BYTE v49[16]; // [rsp+E0h] [rbp-90h] BYREF
  void (__fastcall *v50)(_BYTE *, _BYTE *, __int64); // [rsp+F0h] [rbp-80h]
  unsigned __int8 (__fastcall *v51)(_BYTE *, _BYTE *); // [rsp+F8h] [rbp-78h]
  __m128i v52; // [rsp+100h] [rbp-70h] BYREF
  __m128i v53; // [rsp+110h] [rbp-60h] BYREF
  _BYTE v54[16]; // [rsp+120h] [rbp-50h] BYREF
  void (__fastcall *v55)(_BYTE *, _BYTE *, __int64); // [rsp+130h] [rbp-40h]
  __int64 v56; // [rsp+138h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 40);
  v37 = a1;
  v32 = (__int64)a2;
  v31 = *(unsigned __int8 **)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  v30 = sub_AA54C0(v4);
  if ( !v30 )
    return 0;
  v5 = v4 + 48;
  v6 = *(_QWORD *)(v4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v4 + 48 == v6
    || !v6
    || (v36 = (unsigned __int8 *)(v6 - 24), (unsigned int)*(unsigned __int8 *)(v6 - 24) - 30 > 0xA) )
  {
LABEL_73:
    BUG();
  }
  if ( *(_BYTE *)(v6 - 24) != 31 || (*(_DWORD *)(v6 - 20) & 0x7FFFFFF) != 1 )
    return 0;
  v29 = *(_QWORD *)(v6 - 56);
  v8 = *(_QWORD *)(v4 + 56);
  if ( v8 == v5 )
    goto LABEL_13;
  v9 = 0;
  do
  {
    v8 = *(_QWORD *)(v8 + 8);
    ++v9;
  }
  while ( v8 != v5 );
  if ( v9 != 2 )
  {
LABEL_13:
    sub_AA72C0(&v47, v4, 1);
    v41 = 0;
    v38 = _mm_loadu_si128(&v47).m128i_u64[0];
    v39 = _mm_loadu_si128(&v48).m128i_u64[0];
    if ( v50 )
    {
      v50(v40, v49, 2);
      v42 = v51;
      v41 = v50;
    }
    v11 = _mm_loadu_si128(&v52);
    v12 = _mm_loadu_si128(&v53);
    v45 = 0;
    v43[0] = v11;
    v43[1] = v12;
    if ( v55 )
    {
      v55(v44, v54, 2);
      v46 = v56;
      v45 = v55;
    }
    while ( 1 )
    {
      v13 = (_BYTE *)v38;
      a2 = (_BYTE *)v38;
      if ( v38 == *(_QWORD *)&v43[0] )
        break;
      while ( 1 )
      {
        if ( !a2 )
          goto LABEL_73;
        v14 = a2 - 24;
        if ( (_BYTE *)a1 != a2 - 24 && v14 != v36 )
        {
          if ( (unsigned int)(unsigned __int8)*(a2 - 24) - 67 > 0xC || !sub_B507F0(v14, v32) )
          {
            if ( v45 )
              v45(v44, v44, 3);
            if ( v41 )
              v41(v40, v40, 3);
            if ( v55 )
              v55(v54, v54, 3);
            if ( v50 )
              v50(v49, v49, 3);
            return 0;
          }
          v13 = (_BYTE *)v38;
        }
        v13 = (_BYTE *)*((_QWORD *)v13 + 1);
        v38 = (unsigned __int64)v13;
        a2 = v13;
        if ( v13 != (_BYTE *)v39 )
          break;
LABEL_30:
        if ( *(_BYTE **)&v43[0] == a2 )
          goto LABEL_31;
      }
      while ( 1 )
      {
        if ( a2 )
          a2 -= 24;
        if ( !v41 )
          sub_4263D6(v14, a2, v10);
        v14 = v40;
        if ( v42(v40, a2) )
          break;
        v10 = 0;
        a2 = *(_BYTE **)(v38 + 8);
        v38 = (unsigned __int64)a2;
        v13 = a2;
        if ( (_BYTE *)v39 == a2 )
          goto LABEL_30;
      }
    }
LABEL_31:
    if ( v45 )
    {
      a2 = v44;
      v45(v44, v44, 3);
    }
    if ( v41 )
    {
      a2 = v40;
      v41(v40, v40, 3);
    }
    if ( v55 )
    {
      a2 = v54;
      v55(v54, v54, 3);
    }
    if ( v50 )
    {
      a2 = v49;
      v50(v49, v49, 3);
    }
  }
  v33 = *(_QWORD *)(v30 + 48);
  v15 = v33 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v33 & 0xFFFFFFFFFFFFFFF8LL) == v30 + 48 )
    goto LABEL_72;
  if ( !v15 )
    goto LABEL_73;
  v34 = v33 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (unsigned int)*(unsigned __int8 *)(v15 - 24) - 30 > 0xA )
  {
LABEL_72:
    sub_BD3990(v31, (__int64)a2);
    goto LABEL_73;
  }
  v16 = sub_BD3990(v31, (__int64)a2);
  if ( *(_BYTE *)(v34 - 24) != 31 )
    return 0;
  if ( (*(_DWORD *)(v34 - 20) & 0x7FFFFFF) != 3 )
    return 0;
  v17 = *(_BYTE **)(v34 - 120);
  if ( *v17 != 82 )
    return 0;
  v18 = (unsigned __int8 *)*((_QWORD *)v17 - 8);
  if ( v16 != v18 && v31 != v18 )
    return 0;
  if ( !(unsigned __int8)sub_F08D10(*((_QWORD *)v17 - 4)) )
    return 0;
  v19 = sub_B53900((__int64)v17);
  v20 = v34;
  if ( !*(_QWORD *)(v34 - 56) )
    return 0;
  v21 = *(_QWORD *)(v34 - 88);
  if ( !v21 )
    return 0;
  if ( v19 == 32 )
  {
    v21 = *(_QWORD *)(v34 - 56);
  }
  else if ( v19 != 33 )
  {
    return 0;
  }
  if ( v21 != v29 )
    return 0;
  v22 = *(_QWORD *)(v4 + 56);
  if ( v22 != v5 )
  {
    v23 = *(_QWORD *)(v22 + 8);
    v24 = (unsigned __int8 *)(v22 - 24);
    if ( v24 != v36 )
    {
      while ( 1 )
      {
        LOWORD(v2) = 0;
        v35 = v20;
        sub_B44500(v24, v20, v2);
        v20 = v35;
        if ( v5 == v23 )
          break;
        v24 = (unsigned __int8 *)(v23 - 24);
        if ( (unsigned __int8 *)(v23 - 24) == v36 )
          break;
        v23 = *(_QWORD *)(v23 + 8);
      }
    }
  }
  *(_QWORD *)&v43[0] = *(_QWORD *)(a1 + 72);
  v25 = (__int64 *)sub_BD5C60(a1);
  *(_QWORD *)&v43[0] = sub_A7B980((__int64 *)v43, v25, 1, 43);
  v47.m128i_i64[0] = sub_A747F0(v43, 1, 90);
  if ( v47.m128i_i64[0] )
  {
    v26 = sub_A71DE0(v47.m128i_i64);
    v27 = (__int64 *)sub_BD5C60(a1);
    *(_QWORD *)&v43[0] = sub_A7B980((__int64 *)v43, v27, 1, 90);
    v28 = (__int64 *)sub_BD5C60(a1);
    *(_QWORD *)&v43[0] = sub_A7B5D0((__int64 *)v43, v28, 0, v26);
  }
  *(_QWORD *)(a1 + 72) = *(_QWORD *)&v43[0];
  return v37;
}
