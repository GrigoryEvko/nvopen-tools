// Function: sub_259E220
// Address: 0x259e220
//
_BOOL8 __fastcall sub_259E220(__int64 a1, __int64 a2)
{
  __int64 *v2; // r13
  char v5; // al
  __int64 *v6; // rdi
  unsigned __int64 v7; // r14
  unsigned __int8 *v8; // r10
  __int64 v9; // rax
  char v10; // al
  __int16 v11; // dx
  __int16 v12; // cx
  __int16 v13; // ax
  unsigned __int8 **v15; // rdx
  unsigned __int8 **v16; // r8
  char v17; // si
  unsigned __int8 v18; // al
  unsigned __int64 v19; // rax
  __int16 v20; // dx
  __int16 v21; // ax
  char v22; // [rsp+16h] [rbp-DAh]
  char v23; // [rsp+17h] [rbp-D9h]
  unsigned __int8 **v24; // [rsp+20h] [rbp-D0h]
  unsigned __int8 *v25; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v26; // [rsp+30h] [rbp-C0h]
  unsigned __int8 **v27; // [rsp+38h] [rbp-B8h]
  bool v28; // [rsp+4Dh] [rbp-A3h] BYREF
  char v29; // [rsp+4Eh] [rbp-A2h] BYREF
  char v30; // [rsp+4Fh] [rbp-A1h] BYREF
  void *v31; // [rsp+50h] [rbp-A0h] BYREF
  int v32; // [rsp+58h] [rbp-98h]
  __m128i v33; // [rsp+60h] [rbp-90h] BYREF
  __m128i v34; // [rsp+70h] [rbp-80h] BYREF
  __m128i v35; // [rsp+80h] [rbp-70h] BYREF
  _QWORD v36[12]; // [rsp+90h] [rbp-60h] BYREF

  v2 = (__int64 *)(a1 + 72);
  v5 = sub_2509800((_QWORD *)(a1 + 72));
  v6 = (__int64 *)(a1 + 72);
  if ( (unsigned int)(v5 - 6) <= 1 )
    v7 = sub_250C680(v6);
  else
    v7 = sub_250D070(v6);
  if ( !v7 )
    goto LABEL_19;
  v8 = (unsigned int)((char)sub_2509800(v2) - 6) <= 1 ? sub_250CBE0(v2, a2) : (unsigned __int8 *)sub_25096F0(v2);
  if ( !v8 )
    goto LABEL_19;
  v26 = (unsigned __int64)v8;
  v32 = 458752;
  v31 = &unk_4A16F38;
  sub_250D230((unsigned __int64 *)&v35, (unsigned __int64)v8, 4, 0);
  v33 = _mm_loadu_si128(&v35);
  if ( (unsigned __int8)sub_252A800(a2, &v33, a1, &v28) )
  {
    v32 |= 0x10001u;
    if ( v28 )
      *(_DWORD *)(a1 + 96) |= 0x10001u;
  }
  if ( !(unsigned __int8)sub_259E0A0(a2, a1, &v33, 1, &v29, 0, 0) )
    goto LABEL_16;
  v9 = **(_QWORD **)(*(_QWORD *)(v26 + 24) + 16LL);
  v30 = 0;
  v22 = *(_BYTE *)(v9 + 8);
  if ( v22 == 7 )
  {
LABEL_12:
    v10 = v32;
    v32 |= 0x40004u;
    if ( (v10 & 1) == 0 )
    {
      if ( !v29 || v22 != 7 && v30 )
        goto LABEL_16;
      v20 = *(_WORD *)(a1 + 96);
      v21 = *(_WORD *)(a1 + 98) | 4;
      *(_WORD *)(a1 + 98) = v21;
      if ( (v20 & 1) == 0 )
      {
        *(_WORD *)(a1 + 96) = v20 | 4;
        goto LABEL_16;
      }
      *(_WORD *)(a1 + 96) = v21;
    }
    return 1;
  }
  v35.m128i_i64[0] = (__int64)v36;
  v35.m128i_i64[1] = 0x300000000LL;
  sub_250D230((unsigned __int64 *)&v34, v26, 2, 0);
  v23 = sub_2526B50(a2, &v34, a1, (__int64)&v35, 1u, &v30, 1u);
  if ( v23 )
  {
    v15 = (unsigned __int8 **)v35.m128i_i64[0];
    v16 = (unsigned __int8 **)(v35.m128i_i64[0] + 16LL * v35.m128i_u32[2]);
    if ( (unsigned __int8 **)v35.m128i_i64[0] != v16 )
    {
      v17 = 0;
      do
      {
        v18 = **v15;
        if ( v18 <= 0x15u )
        {
          if ( v17 )
            goto LABEL_31;
          v17 = v23;
        }
        else
        {
          v24 = v16;
          v25 = *v15;
          v27 = v15;
          if ( v18 != 22 )
            goto LABEL_31;
          v19 = sub_250C680(v2);
          v15 = v27;
          v16 = v24;
          if ( v25 == (unsigned __int8 *)v19 )
            goto LABEL_31;
        }
        v15 += 2;
      }
      while ( v16 != v15 );
      v16 = (unsigned __int8 **)v35.m128i_i64[0];
    }
    if ( v16 != v36 )
      _libc_free((unsigned __int64)v16);
    goto LABEL_12;
  }
LABEL_31:
  if ( (_QWORD *)v35.m128i_i64[0] != v36 )
    _libc_free(v35.m128i_u64[0]);
LABEL_16:
  v35.m128i_i64[0] = (__int64)&v34;
  v34.m128i_i64[0] = a2;
  v34.m128i_i64[1] = a1;
  v35.m128i_i64[1] = a2;
  v36[0] = &v31;
  v36[1] = a1;
  if ( !(unsigned __int8)sub_252FFB0(
                           a2,
                           (unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *))sub_2589160,
                           (__int64)&v35,
                           a1,
                           v7,
                           0,
                           1,
                           1,
                           0,
                           0) )
  {
LABEL_19:
    *(_WORD *)(a1 + 98) = *(_WORD *)(a1 + 96);
    return 0;
  }
  v11 = *(_WORD *)(a1 + 98);
  v12 = *(_WORD *)(a1 + 96);
  v13 = v12 | v11 & HIWORD(v32);
  if ( (((unsigned __int8)v12 | (unsigned __int8)(v11 & BYTE2(v32))) & 3) == 3 )
  {
    *(_WORD *)(a1 + 98) = v13;
    return v13 == v11;
  }
  else
  {
    *(_WORD *)(a1 + 98) = v12;
    return 0;
  }
}
