// Function: sub_3113440
// Address: 0x3113440
//
_QWORD *__fastcall sub_3113440(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  unsigned __int64 v7; // r14
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rbx
  __m128i *v11; // rdi
  __int64 v13; // rax
  unsigned __int64 v14; // rax
  __int64 (__fastcall *v15)(__int64); // rax
  char v16; // al
  unsigned __int64 *v17; // r8
  __int64 v18; // rax
  __m128i v19; // xmm1
  __int64 (__fastcall *v20)(__int64); // rax
  char v21; // al
  unsigned __int64 *v22; // [rsp+8h] [rbp-98h]
  unsigned __int64 *v23; // [rsp+8h] [rbp-98h]
  unsigned __int64 v24; // [rsp+18h] [rbp-88h] BYREF
  __int64 v25; // [rsp+20h] [rbp-80h] BYREF
  __int64 v26; // [rsp+28h] [rbp-78h]
  _BYTE v27[16]; // [rsp+30h] [rbp-70h] BYREF
  __m128i v28; // [rsp+40h] [rbp-60h] BYREF
  __m128i v29; // [rsp+50h] [rbp-50h] BYREF
  __int16 v30; // [rsp+60h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 64);
  v7 = *(_QWORD *)(v6 + 16);
  v8 = *(_QWORD *)(v6 + 8);
  if ( v7 - v8 <= 0x17 )
  {
    *(_DWORD *)(a2 + 8) = 3;
    v26 = 0;
    v25 = (__int64)v27;
    v27[0] = 0;
    sub_2240AE0((unsigned __int64 *)(a2 + 16), (unsigned __int64 *)&v25);
    v28.m128i_i64[0] = (__int64)&v25;
    v30 = 260;
    v9 = sub_22077B0(0x30u);
    v10 = v9;
    if ( v9 )
    {
      *(_DWORD *)(v9 + 8) = 3;
      *(_QWORD *)v9 = &unk_4A32A78;
      sub_CA0F50((__int64 *)(v9 + 16), (void **)&v28);
    }
    goto LABEL_4;
  }
  sub_3111DC0((__int64)&v28, v8, a3, a4, a5);
  if ( (v30 & 1) != 0 )
  {
    LOBYTE(v30) = v30 & 0xFD;
    v13 = v28.m128i_i64[0];
    v28.m128i_i64[0] = 0;
    v25 = v13 | 1;
    v14 = v13 & 0xFFFFFFFFFFFFFFFELL;
    if ( v14 )
    {
      *a1 = v14 | 1;
      return a1;
    }
  }
  else
  {
    v25 = 1;
    v19 = _mm_loadu_si128(&v29);
    *(__m128i *)(a2 + 72) = _mm_loadu_si128(&v28);
    *(__m128i *)(a2 + 88) = v19;
  }
  v15 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 40LL);
  if ( v15 == sub_3112820 )
    v16 = *(_BYTE *)(a2 + 84) & 1;
  else
    v16 = v15(a2);
  v17 = (unsigned __int64 *)(a2 + 16);
  if ( v16 )
  {
    v24 = v8 + *(_QWORD *)(a2 + 88);
    if ( v24 >= v7 )
      goto LABEL_13;
    sub_31172F0(a2 + 48, &v24);
    v17 = (unsigned __int64 *)(a2 + 16);
  }
  v20 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 48LL);
  if ( v20 == sub_3112830 )
  {
    if ( (*(_BYTE *)(a2 + 84) & 2) == 0 )
      goto LABEL_22;
  }
  else
  {
    v23 = v17;
    v21 = v20(a2);
    v17 = v23;
    if ( !v21 )
    {
LABEL_22:
      v29.m128i_i8[0] = 0;
      *(_DWORD *)(a2 + 8) = 0;
      v28 = (__m128i)(unsigned __int64)&v29;
      sub_2240AE0(v17, (unsigned __int64 *)&v28);
      v11 = (__m128i *)v28.m128i_i64[0];
      *a1 = 1;
      if ( v11 == &v29 )
        return a1;
      goto LABEL_5;
    }
  }
  v24 = *(_QWORD *)(a2 + 96) + v8;
  if ( v24 < v7 )
  {
    v22 = v17;
    sub_311F0C0(a2 + 56, &v24);
    v17 = v22;
    goto LABEL_22;
  }
LABEL_13:
  v27[0] = 0;
  *(_DWORD *)(a2 + 8) = 1;
  v25 = (__int64)v27;
  v26 = 0;
  sub_2240AE0(v17, (unsigned __int64 *)&v25);
  v28.m128i_i64[0] = (__int64)&v25;
  v30 = 260;
  v18 = sub_22077B0(0x30u);
  v10 = v18;
  if ( v18 )
  {
    *(_DWORD *)(v18 + 8) = 1;
    *(_QWORD *)v18 = &unk_4A32A78;
    sub_CA0F50((__int64 *)(v18 + 16), (void **)&v28);
  }
LABEL_4:
  v11 = (__m128i *)v25;
  *a1 = v10 | 1;
  if ( v11 != (__m128i *)v27 )
LABEL_5:
    j_j___libc_free_0((unsigned __int64)v11);
  return a1;
}
