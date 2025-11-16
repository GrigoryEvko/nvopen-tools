// Function: sub_38147C0
// Address: 0x38147c0
//
unsigned __int8 *__fastcall sub_38147C0(__int64 *a1, __int64 a2, __m128i a3)
{
  __int16 *v4; // rax
  __int64 v5; // rsi
  unsigned __int16 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rdx
  int v9; // r9d
  __int64 v10; // r10
  __int64 v11; // r15
  __int64 (__fastcall *v12)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v13; // r8
  __int64 v14; // rcx
  unsigned int v15; // r9d
  unsigned __int8 *v16; // r12
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  int v22; // [rsp+Ch] [rbp-94h]
  __int64 v23; // [rsp+10h] [rbp-90h] BYREF
  __int64 v24; // [rsp+18h] [rbp-88h]
  __int64 v25; // [rsp+20h] [rbp-80h] BYREF
  int v26; // [rsp+28h] [rbp-78h]
  __int64 v27; // [rsp+30h] [rbp-70h]
  __int64 v28; // [rsp+38h] [rbp-68h]
  __int64 v29; // [rsp+40h] [rbp-60h]
  __int64 v30; // [rsp+48h] [rbp-58h]
  char v31[8]; // [rsp+50h] [rbp-50h] BYREF
  unsigned __int16 v32; // [rsp+58h] [rbp-48h]
  __int64 v33; // [rsp+60h] [rbp-40h]

  v4 = *(__int16 **)(a2 + 48);
  v5 = *(_QWORD *)(a2 + 80);
  v6 = *v4;
  v7 = *((_QWORD *)v4 + 1);
  v25 = v5;
  LOWORD(v23) = v6;
  v24 = v7;
  if ( v5 )
    sub_B96E90((__int64)&v25, v5, 1);
  v26 = *(_DWORD *)(a2 + 72);
  if ( v6 )
  {
    if ( v6 == 1 || (unsigned __int16)(v6 - 504) <= 7u )
      BUG();
    v18 = *(_QWORD *)&byte_444C4A0[16 * v6 - 16];
    if ( !v18 )
      goto LABEL_5;
  }
  else
  {
    v27 = sub_3007260((__int64)&v23);
    v28 = v8;
    if ( !v27 )
    {
LABEL_5:
      v9 = 214;
      goto LABEL_6;
    }
    v18 = sub_3007260((__int64)&v23);
    v29 = v18;
    v30 = v19;
  }
  v9 = ((v18 & 7) != 0) + 213;
LABEL_6:
  v10 = *a1;
  v11 = a1[1];
  v22 = v9;
  v12 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v12 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)v31, v10, *(_QWORD *)(v11 + 64), v23, v24);
    v13 = v33;
    v14 = v32;
    v15 = v22;
  }
  else
  {
    v20 = v12(v10, *(_QWORD *)(v11 + 64), v23, v24);
    v15 = v22;
    v14 = v20;
    v13 = v21;
  }
  v16 = sub_33FAF80(v11, v15, (__int64)&v25, v14, v13, v15, a3);
  if ( v25 )
    sub_B91220((__int64)&v25, v25);
  return v16;
}
