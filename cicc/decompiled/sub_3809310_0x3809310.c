// Function: sub_3809310
// Address: 0x3809310
//
unsigned __int8 *__fastcall sub_3809310(_QWORD *a1, unsigned __int64 a2)
{
  int v3; // eax
  __int64 v4; // rdx
  char v5; // r9
  __int64 v6; // rsi
  __m128i v7; // xmm0
  __int64 v8; // rax
  __int16 v9; // dx
  __int16 *v10; // rax
  unsigned __int16 v11; // cx
  __int64 v12; // rbx
  unsigned int v13; // r15d
  __int32 v14; // edx
  __int64 *v15; // rax
  __int64 v16; // r8
  __int64 v17; // r9
  _WORD *v18; // r10
  __int64 v19; // r11
  int v20; // r9d
  unsigned __int8 *v21; // r9
  __int64 v22; // rdx
  __int64 v23; // r15
  unsigned __int8 *v24; // r12
  unsigned __int64 v26; // rsi
  char v27; // [rsp+0h] [rbp-E0h]
  int v28; // [rsp+0h] [rbp-E0h]
  unsigned __int16 v29; // [rsp+6h] [rbp-DAh]
  bool v30; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v31; // [rsp+8h] [rbp-D8h]
  __m128i v32; // [rsp+20h] [rbp-C0h] BYREF
  int v33; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v34; // [rsp+38h] [rbp-A8h]
  unsigned int v35; // [rsp+40h] [rbp-A0h] BYREF
  __int64 (__fastcall *v36)(__int64, __int64, unsigned int); // [rsp+48h] [rbp-98h]
  __int64 v37; // [rsp+50h] [rbp-90h] BYREF
  int v38; // [rsp+58h] [rbp-88h]
  _QWORD v39[6]; // [rsp+60h] [rbp-80h] BYREF
  __int64 v40; // [rsp+90h] [rbp-50h]
  __int64 v41; // [rsp+98h] [rbp-48h]
  __int64 v42; // [rsp+A0h] [rbp-40h]

  v3 = *(_DWORD *)(a2 + 24);
  if ( v3 > 239 )
  {
    v4 = (unsigned int)(v3 - 242) < 2 ? 0x28 : 0;
    v30 = (unsigned int)(v3 - 242) < 2;
LABEL_7:
    v5 = v3 == 141;
    goto LABEL_8;
  }
  if ( v3 <= 237 && (unsigned int)(v3 - 101) > 0x2F )
  {
    v4 = 0;
    if ( v3 == 226 )
    {
      v30 = 0;
      v5 = 1;
      goto LABEL_8;
    }
    v30 = 0;
    goto LABEL_7;
  }
  v4 = 40;
  if ( v3 != 226 )
  {
    v30 = 1;
    goto LABEL_7;
  }
  v30 = 1;
  v5 = 1;
LABEL_8:
  v6 = *(_QWORD *)(a2 + 80);
  v7 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + v4));
  v32 = v7;
  v8 = *(_QWORD *)(v7.m128i_i64[0] + 48) + 16LL * v7.m128i_u32[2];
  v9 = *(_WORD *)v8;
  v34 = *(_QWORD *)(v8 + 8);
  v10 = *(__int16 **)(a2 + 48);
  LOWORD(v33) = v9;
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  v37 = v6;
  v36 = 0;
  v29 = v11;
  v13 = v11;
  LOWORD(v35) = 0;
  if ( v6 )
  {
    v27 = v5;
    sub_B96E90((__int64)&v37, v6, 1);
    v5 = v27;
  }
  v38 = *(_DWORD *)(a2 + 72);
  v28 = sub_37FBE80(v33, v34, v13, v12, (__int64)&v35, v5);
  v32.m128i_i64[0] = sub_3805E70((__int64)a1, v32.m128i_u64[0], v32.m128i_i64[1]);
  v32.m128i_i32[2] = v14;
  if ( v30 )
  {
    v15 = *(__int64 **)(a2 + 40);
    v16 = *v15;
    v17 = v15[1];
  }
  else
  {
    v16 = 0;
    v17 = 0;
  }
  v18 = (_WORD *)*a1;
  v19 = a1[1];
  LOWORD(v40) = v29;
  v39[5] = 1;
  v41 = v12;
  LOBYTE(v42) = 20;
  sub_3494590(
    (__int64)v39,
    v18,
    v19,
    v28,
    v35,
    v36,
    (__int64)&v32,
    1u,
    (__int64)&v33,
    1,
    v40,
    v12,
    20,
    (__int64)&v37,
    v16,
    v17);
  v21 = sub_33FAF80(a1[1], 216, (__int64)&v37, v13, v12, v20, v7);
  v23 = v22;
  if ( v30 )
  {
    v31 = (unsigned __int64)v21;
    sub_3760E70((__int64)a1, a2, 1, v39[2], v39[3]);
    v26 = a2;
    v24 = 0;
    sub_3760E70((__int64)a1, v26, 0, v31, v23);
  }
  else
  {
    v24 = v21;
  }
  if ( v37 )
    sub_B91220((__int64)&v37, v37);
  return v24;
}
