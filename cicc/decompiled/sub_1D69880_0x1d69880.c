// Function: sub_1D69880
// Address: 0x1d69880
//
__int64 __fastcall sub_1D69880(_BYTE *a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 (__fastcall *v7)(__int64); // r8
  unsigned int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 (*v12)(); // rax
  char v14; // al
  int v15; // eax
  __int64 *v16; // rsi
  __int64 v17; // rdx
  char v18; // bl
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rdx
  char v23; // al
  int v24; // r13d
  char v25; // r15
  __int64 v26; // rdx
  char v27; // al
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  unsigned int v35; // r15d
  unsigned int v36; // eax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // [rsp+0h] [rbp-70h] BYREF
  __int64 v40; // [rsp+8h] [rbp-68h]
  __m128i v41; // [rsp+10h] [rbp-60h] BYREF
  __m128i v42; // [rsp+20h] [rbp-50h] BYREF
  __int64 v43; // [rsp+30h] [rbp-40h]

  if ( a1[16] != 72 )
    goto LABEL_10;
  v5 = *(_QWORD *)a1;
  v6 = *a2;
  v7 = *(__int64 (__fastcall **)(__int64))(*a2 + 584);
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16 )
    v5 = **(_QWORD **)(v5 + 16);
  v8 = *(_DWORD *)(v5 + 8);
  v9 = **((_QWORD **)a1 - 3);
  v10 = v8 >> 8;
  if ( *(_BYTE *)(v9 + 8) == 16 )
  {
    v11 = *(_DWORD *)(**(_QWORD **)(v9 + 16) + 8LL) >> 8;
    if ( v7 == sub_1D5A7B0 )
      goto LABEL_6;
  }
  else
  {
    v11 = *(_DWORD *)(v9 + 8) >> 8;
    if ( v7 == sub_1D5A7B0 )
    {
LABEL_6:
      v12 = *(__int64 (**)())(v6 + 576);
      if ( v12 == sub_1D12D90 )
        return 0;
      v14 = ((__int64 (__fastcall *)(__int64 *, __int64, __int64))v12)(a2, v11, v10);
      goto LABEL_9;
    }
  }
  v14 = ((__int64 (__fastcall *)(__int64 *, __int64, __int64))v7)(a2, v11, v10);
LABEL_9:
  if ( !v14 )
    return 0;
LABEL_10:
  v15 = sub_1D5D7E0(a3, **((__int64 ***)a1 - 3), 0);
  v16 = *(__int64 **)a1;
  v40 = v17;
  LODWORD(v39) = v15;
  v18 = v15;
  v41.m128i_i32[0] = sub_1D5D7E0(a3, v16, 0);
  v41.m128i_i64[1] = v22;
  if ( (_BYTE)v39 )
  {
    v24 = v41.m128i_u8[0];
    v25 = (unsigned __int8)(v18 - 14) <= 0x47u || (unsigned __int8)(v18 - 2) <= 5u;
    if ( v41.m128i_i8[0] )
      goto LABEL_12;
  }
  else
  {
    v23 = sub_1F58CF0(&v39);
    v24 = v41.m128i_u8[0];
    v25 = v23;
    if ( v41.m128i_i8[0] )
    {
LABEL_12:
      v26 = (unsigned int)(v24 - 14);
      LOBYTE(v26) = (unsigned __int8)(v24 - 14) <= 0x47u;
      v27 = v26 | ((unsigned __int8)(v24 - 2) <= 5u);
      goto LABEL_13;
    }
  }
  v27 = sub_1F58CF0(&v41);
LABEL_13:
  if ( v27 != v25 )
    return 0;
  v42 = _mm_loadu_si128(&v41);
  if ( (_BYTE)v24 == v18 )
  {
    if ( (_BYTE)v24 || v42.m128i_i64[1] == v40 )
      goto LABEL_16;
LABEL_33:
    v35 = sub_1F58D40(&v39, v16, v26, v19, v20, v21);
    if ( !(_BYTE)v24 )
      goto LABEL_34;
LABEL_27:
    v36 = sub_1D5A920(v24);
    goto LABEL_28;
  }
  if ( !v18 )
    goto LABEL_33;
  v35 = sub_1D5A920(v18);
  if ( (_BYTE)v24 )
    goto LABEL_27;
LABEL_34:
  v36 = sub_1F58D40(&v42, v16, v31, v32, v33, v34);
LABEL_28:
  if ( v36 > v35 )
    return 0;
LABEL_16:
  v28 = sub_16498A0((__int64)a1);
  sub_1F40D10(&v42, a2, v28, v39, v40);
  if ( v42.m128i_i8[0] == 1 )
  {
    v37 = sub_16498A0((__int64)a1);
    sub_1F40D10(&v42, a2, v37, v39, v40);
    v18 = v42.m128i_i8[8];
    LOBYTE(v39) = v42.m128i_i8[8];
    v40 = v43;
  }
  v29 = sub_16498A0((__int64)a1);
  sub_1F40D10(&v42, a2, v29, v41.m128i_i64[0], v41.m128i_i64[1]);
  if ( v42.m128i_i8[0] == 1 )
  {
    v38 = sub_16498A0((__int64)a1);
    sub_1F40D10(&v42, a2, v38, v41.m128i_i64[0], v41.m128i_i64[1]);
    LOBYTE(v24) = v42.m128i_i8[8];
    v30 = v43;
    v41.m128i_i8[0] = v42.m128i_i8[8];
    v41.m128i_i64[1] = v43;
  }
  else
  {
    v30 = v41.m128i_i64[1];
  }
  if ( (_BYTE)v24 != v18 || !(_BYTE)v24 && v40 != v30 )
    return 0;
  return sub_1D69330((__int64)a1);
}
