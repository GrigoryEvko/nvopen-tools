// Function: sub_3808DE0
// Address: 0x3808de0
//
unsigned __int64 __fastcall sub_3808DE0(_QWORD *a1, unsigned __int64 a2)
{
  int v3; // eax
  __int64 v4; // rdx
  unsigned int v5; // eax
  __int64 v6; // rdx
  __int16 v7; // cx
  __int64 v8; // rdx
  __int64 (__fastcall *v9)(__int64, __int64, unsigned int); // rbx
  __int16 v10; // dx
  int v11; // ecx
  __int64 *v12; // rax
  __int64 v13; // r14
  __int64 v14; // r15
  unsigned __int64 v15; // rax
  __int64 v16; // rsi
  _WORD *v17; // r10
  int v18; // ecx
  __int32 v19; // edx
  __int64 v20; // rsi
  _WORD *v22; // [rsp+8h] [rbp-D8h]
  int v23; // [rsp+10h] [rbp-D0h]
  unsigned __int16 v24; // [rsp+1Ch] [rbp-C4h]
  bool v25; // [rsp+1Fh] [rbp-C1h]
  __m128i v26; // [rsp+30h] [rbp-B0h] BYREF
  int v27; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v28; // [rsp+48h] [rbp-98h]
  __int64 v29; // [rsp+50h] [rbp-90h] BYREF
  int v30; // [rsp+58h] [rbp-88h]
  unsigned __int64 v31[4]; // [rsp+60h] [rbp-80h] BYREF
  int *v32; // [rsp+80h] [rbp-60h]
  __int64 v33; // [rsp+88h] [rbp-58h]
  __int64 v34; // [rsp+90h] [rbp-50h]
  __int64 (__fastcall *v35)(__int64, __int64, unsigned int); // [rsp+98h] [rbp-48h]
  __int64 v36; // [rsp+A0h] [rbp-40h]

  v3 = *(_DWORD *)(a2 + 24);
  if ( v3 > 239 )
  {
    v4 = (unsigned int)(v3 - 242) < 2 ? 0x28 : 0;
    v25 = (unsigned int)(v3 - 242) < 2;
  }
  else if ( v3 > 237 )
  {
    v25 = 1;
    v4 = 40;
  }
  else
  {
    v4 = (unsigned int)(v3 - 101) < 0x30 ? 0x28 : 0;
    v25 = (unsigned int)(v3 - 101) < 0x30;
  }
  v5 = v3 & 0xFFFFFFFD;
  v26 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + v4));
  v6 = *(_QWORD *)(v26.m128i_i64[0] + 48) + 16LL * v26.m128i_u32[2];
  v7 = *(_WORD *)v6;
  v28 = *(_QWORD *)(v6 + 8);
  v8 = *(_QWORD *)(a2 + 48);
  LOWORD(v27) = v7;
  v9 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(v8 + 8);
  v24 = *(_WORD *)v8;
  v10 = *(_WORD *)v8;
  if ( v5 == 237 )
  {
    v10 = 11;
  }
  else if ( v5 == 241 )
  {
    v10 = 10;
  }
  v11 = sub_2FE5850(v27, v28, v10);
  if ( v25 )
  {
    v12 = *(__int64 **)(a2 + 40);
    v13 = *v12;
    v14 = v12[1];
  }
  else
  {
    v13 = 0;
    v14 = 0;
  }
  v23 = v11;
  v15 = sub_3805E70((__int64)a1, v26.m128i_u64[0], v26.m128i_i64[1]);
  v16 = *(_QWORD *)(a2 + 80);
  v35 = v9;
  v17 = (_WORD *)*a1;
  v18 = v23;
  v26.m128i_i64[0] = v15;
  v33 = 1;
  v26.m128i_i32[2] = v19;
  v32 = &v27;
  LOBYTE(v36) = 20;
  LOWORD(v34) = v24;
  v29 = v16;
  if ( v16 )
  {
    v22 = v17;
    sub_B96E90((__int64)&v29, v16, 1);
    v18 = v23;
    v17 = v22;
  }
  v20 = a1[1];
  v30 = *(_DWORD *)(a2 + 72);
  sub_3494590(
    (__int64)v31,
    v17,
    v20,
    v18,
    v24,
    v9,
    (__int64)&v26,
    1u,
    (__int64)v32,
    v33,
    v34,
    (__int64)v35,
    v36,
    (__int64)&v29,
    v13,
    v14);
  if ( v29 )
    sub_B91220((__int64)&v29, v29);
  if ( !v25 )
    return v31[0];
  sub_3760E70((__int64)a1, a2, 1, v31[2], v31[3]);
  sub_3760E70((__int64)a1, a2, 0, v31[0], v31[1]);
  return 0;
}
