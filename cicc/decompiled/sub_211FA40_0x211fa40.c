// Function: sub_211FA40
// Address: 0x211fa40
//
__int64 __fastcall sub_211FA40(__int64 *a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v7; // rax
  __int64 v8; // rsi
  char v9; // di
  __int64 v10; // rax
  unsigned int v11; // eax
  __int64 v12; // r15
  unsigned int v13; // esi
  bool v14; // cc
  char v15; // al
  char v16; // r14
  __int64 *v17; // rsi
  __int64 v18; // r14
  __int64 v19; // rdx
  __int64 v20; // r15
  __int64 v21; // rsi
  __int64 v22; // r14
  const void **v24; // rdx
  unsigned int v25; // eax
  __int128 v26; // [rsp-10h] [rbp-90h]
  const void **v27; // [rsp+8h] [rbp-78h]
  __int64 v28; // [rsp+10h] [rbp-70h] BYREF
  __int64 v29; // [rsp+18h] [rbp-68h]
  __int64 v30; // [rsp+20h] [rbp-60h] BYREF
  int v31; // [rsp+28h] [rbp-58h]
  __int64 v32; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v33; // [rsp+38h] [rbp-48h]
  const void **v34; // [rsp+40h] [rbp-40h]

  v7 = *(_QWORD *)(a2 + 40);
  v8 = *(_QWORD *)(a2 + 72);
  v9 = *(_BYTE *)v7;
  v10 = *(_QWORD *)(v7 + 8);
  v30 = v8;
  LOBYTE(v28) = v9;
  v29 = v10;
  if ( v8 )
  {
    sub_1623A60((__int64)&v30, v8, 2);
    v9 = v28;
  }
  v31 = *(_DWORD *)(a2 + 64);
  if ( v9 )
  {
    v11 = sub_211A7A0(v9);
    v12 = a1[1];
    v13 = v11;
    v14 = v11 <= 0x20;
    if ( v11 != 32 )
      goto LABEL_5;
LABEL_25:
    v15 = 5;
    goto LABEL_8;
  }
  v25 = sub_1F58D40((__int64)&v28);
  v12 = a1[1];
  v13 = v25;
  v14 = v25 <= 0x20;
  if ( v25 == 32 )
    goto LABEL_25;
LABEL_5:
  if ( v14 )
  {
    if ( v13 == 8 )
    {
      v15 = 3;
    }
    else
    {
      v15 = 4;
      if ( v13 != 16 )
      {
        v15 = 2;
        if ( v13 != 1 )
          goto LABEL_20;
      }
    }
LABEL_8:
    v27 = 0;
    goto LABEL_9;
  }
  if ( v13 == 64 )
  {
    v15 = 6;
    goto LABEL_8;
  }
  if ( v13 == 128 )
  {
    v15 = 7;
    goto LABEL_8;
  }
LABEL_20:
  v15 = sub_1F58CC0(*(_QWORD **)(v12 + 48), v13);
  v12 = a1[1];
  v27 = v24;
LABEL_9:
  v16 = v15;
  v17 = (__int64 *)(*(_QWORD *)(a2 + 88) + 32LL);
  if ( (void *)*v17 == sub_16982C0() )
    sub_169D930((__int64)&v32, (__int64)v17);
  else
    sub_169D7E0((__int64)&v32, v17);
  v18 = sub_1D38970(v12, (__int64)&v32, (__int64)&v30, v16, v27, 0, a3, a4, a5, 0);
  v20 = v19;
  if ( v33 > 0x40 && v32 )
    j_j___libc_free_0_0(v32);
  sub_1F40D10((__int64)&v32, *a1, *(_QWORD *)(a1[1] + 48), v28, v29);
  if ( (_BYTE)v28 == 8 )
  {
    v21 = 160;
  }
  else
  {
    if ( (_BYTE)v33 != 8 )
      sub_16BD130("Attempt at an invalid promotion-related conversion", 1u);
    v21 = 161;
  }
  *((_QWORD *)&v26 + 1) = v20;
  *(_QWORD *)&v26 = v18;
  v22 = sub_1D309E0(
          (__int64 *)a1[1],
          v21,
          (__int64)&v30,
          (unsigned __int8)v33,
          v34,
          0,
          *(double *)a3.m128i_i64,
          a4,
          *(double *)a5.m128i_i64,
          v26);
  if ( v30 )
    sub_161E7C0((__int64)&v30, v30);
  return v22;
}
