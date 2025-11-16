// Function: sub_206C720
// Address: 0x206c720
//
__int64 *__fastcall sub_206C720(__int64 a1, __int64 a2, unsigned int a3, __m128i a4, __m128i a5, __m128i a6)
{
  unsigned int v6; // r13d
  unsigned __int64 v8; // rax
  __int64 v9; // rdx
  unsigned int v10; // eax
  __int64 **v11; // rax
  __int64 *v12; // r14
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // r15
  __int64 v15; // rax
  __int64 *v16; // rax
  __int64 *v17; // r11
  __int64 *v18; // r8
  __int64 v19; // rdx
  __int64 v20; // r9
  unsigned __int8 *v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 *v25; // r13
  int v26; // edx
  int v27; // r14d
  __int64 *result; // rax
  unsigned __int64 v29; // rcx
  void *v30; // rax
  __int128 v31; // [rsp-10h] [rbp-A0h]
  __int64 v32; // [rsp+8h] [rbp-88h]
  __int64 *v33; // [rsp+10h] [rbp-80h]
  __int64 v34; // [rsp+18h] [rbp-78h]
  __int64 *v35; // [rsp+20h] [rbp-70h]
  const void **v37; // [rsp+38h] [rbp-58h]
  __int64 v38; // [rsp+50h] [rbp-40h] BYREF
  int v39; // [rsp+58h] [rbp-38h]

  v6 = 0;
  v8 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)v8 <= 0x17u )
  {
    if ( (_BYTE)v8 == 5 )
    {
      v29 = *(unsigned __int16 *)(a2 + 18);
      if ( (unsigned __int16)v29 <= 0x17u )
      {
        v30 = &loc_80A800;
        if ( _bittest64((const __int64 *)&v30, v29) )
          LOBYTE(v6) = *(_BYTE *)(a2 + 17) & 6 | 1;
      }
      if ( (unsigned int)(v29 - 17) <= 1 || (unsigned __int16)(*(_WORD *)(a2 + 18) - 24) <= 1u )
        goto LABEL_7;
    }
  }
  else
  {
    if ( (unsigned __int8)v8 <= 0x2Fu )
    {
      v9 = 0x80A800000000LL;
      if ( _bittest64(&v9, v8) )
        LOBYTE(v6) = *(_BYTE *)(a2 + 17) & 6 | 1;
    }
    if ( (unsigned __int8)(v8 - 48) <= 1u || (unsigned int)(v8 - 41) <= 1 )
LABEL_7:
      LOBYTE(v6) = v6 & 0xF6 | (4 * *(_BYTE *)(a2 + 17)) & 8 | 1;
  }
  LOBYTE(v10) = sub_2047BD0((_BYTE *)a2);
  HIWORD(v10) = 0;
  if ( (_BYTE)v10 )
  {
    LOWORD(v10) = v6 | 0x101;
    v6 = v10;
  }
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v11 = *(__int64 ***)(a2 - 8);
  else
    v11 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v12 = sub_20685E0(a1, *v11, a4, a5, a6);
  v14 = v13;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v15 = *(_QWORD *)(a2 - 8);
  else
    v15 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v16 = sub_20685E0(a1, *(__int64 **)(v15 + 24), a4, a5, a6);
  v17 = *(__int64 **)(a1 + 552);
  v18 = v16;
  v20 = v19;
  LODWORD(v19) = *(_DWORD *)(a1 + 536);
  v21 = (unsigned __int8 *)(v12[5] + 16LL * (unsigned int)v14);
  v37 = (const void **)*((_QWORD *)v21 + 1);
  v22 = *v21;
  v23 = *(_QWORD *)a1;
  v38 = 0;
  v39 = v19;
  if ( v23 )
  {
    if ( &v38 != (__int64 *)(v23 + 48) )
    {
      v24 = *(_QWORD *)(v23 + 48);
      v38 = v24;
      if ( v24 )
      {
        v32 = v22;
        v33 = v18;
        v34 = v20;
        v35 = v17;
        sub_1623A60((__int64)&v38, v24, 2);
        v22 = v32;
        v18 = v33;
        v20 = v34;
        v17 = v35;
      }
    }
  }
  *((_QWORD *)&v31 + 1) = v20;
  *(_QWORD *)&v31 = v18;
  v25 = sub_1D332F0(
          v17,
          a3,
          (__int64)&v38,
          v22,
          v37,
          v6,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128i_i64,
          a6,
          (__int64)v12,
          v14,
          v31);
  v27 = v26;
  if ( v38 )
    sub_161E7C0((__int64)&v38, v38);
  v38 = a2;
  result = sub_205F5C0(a1 + 8, &v38);
  result[1] = (__int64)v25;
  *((_DWORD *)result + 4) = v27;
  return result;
}
