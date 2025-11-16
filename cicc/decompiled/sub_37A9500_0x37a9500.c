// Function: sub_37A9500
// Address: 0x37a9500
//
__int64 __fastcall sub_37A9500(__int64 a1, unsigned __int64 a2)
{
  __int64 v4; // rax
  int v5; // r14d
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // r8
  __int64 v11; // r9
  __int32 v12; // r11d
  unsigned __int8 v13; // dl
  unsigned __int16 v14; // cx
  unsigned __int64 v15; // rsi
  __m128i v16; // xmm0
  __m128i v17; // xmm1
  _QWORD *v18; // rdi
  unsigned __int16 v19; // ax
  unsigned __int64 v20; // r8
  __m128i *v21; // r14
  const __m128i *v23; // [rsp-20h] [rbp-F0h]
  __int64 v24; // [rsp+0h] [rbp-D0h]
  __int64 v25; // [rsp+8h] [rbp-C8h]
  __int64 v26; // [rsp+18h] [rbp-B8h]
  __int64 v27; // [rsp+20h] [rbp-B0h]
  int v28; // [rsp+28h] [rbp-A8h]
  int v29; // [rsp+2Ch] [rbp-A4h]
  __int64 v30; // [rsp+30h] [rbp-A0h] BYREF
  int v31; // [rsp+38h] [rbp-98h]
  __m128i v32; // [rsp+40h] [rbp-90h] BYREF
  __int64 v33; // [rsp+50h] [rbp-80h]
  int v34; // [rsp+58h] [rbp-78h]
  __int64 v35; // [rsp+60h] [rbp-70h]
  int v36; // [rsp+68h] [rbp-68h]
  __m128i v37; // [rsp+70h] [rbp-60h]
  __int64 v38; // [rsp+80h] [rbp-50h]
  __int64 v39; // [rsp+88h] [rbp-48h]
  __int64 v40; // [rsp+90h] [rbp-40h]
  int v41; // [rsp+98h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 40);
  v29 = *(_DWORD *)(v4 + 88);
  v26 = *(_QWORD *)(v4 + 40);
  v5 = *(_DWORD *)(v4 + 48);
  v27 = *(_QWORD *)(v4 + 200);
  v6 = *(_QWORD *)(v4 + 80);
  v28 = *(_DWORD *)(v4 + 208);
  v7 = sub_379AB60(a1, *(_QWORD *)(v4 + 160), *(_QWORD *)(v4 + 168));
  v9 = *(_QWORD *)(a2 + 80);
  v10 = v7;
  v11 = v8;
  v30 = v9;
  if ( v9 )
  {
    v25 = v8;
    v24 = v7;
    sub_B96E90((__int64)&v30, v9, 1);
    v10 = v24;
    v11 = v25;
  }
  v12 = *(_DWORD *)(a2 + 68);
  v13 = *(_BYTE *)(a2 + 33);
  v31 = *(_DWORD *)(a2 + 72);
  v14 = *(_WORD *)(a2 + 96);
  v15 = *(_QWORD *)(a2 + 48);
  v16 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v17 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 120LL));
  v38 = v10;
  v39 = v11;
  v40 = v27;
  v33 = v26;
  v18 = *(_QWORD **)(a1 + 8);
  v41 = v28;
  v19 = *(_WORD *)(a2 + 32);
  v35 = v6;
  v32 = v16;
  v37 = v17;
  v23 = *(const __m128i **)(a2 + 112);
  v20 = *(_QWORD *)(a2 + 104);
  v34 = v5;
  v36 = v29;
  v21 = sub_33E8420(
          v18,
          v15,
          v12,
          v14,
          v20,
          (__int64)&v30,
          (unsigned __int64 *)&v32,
          6,
          v23,
          (v19 >> 7) & 7,
          (v13 >> 2) & 3);
  sub_3760E70(a1, a2, 1, (unsigned __int64)v21, 1);
  sub_3760E70(a1, a2, 0, (unsigned __int64)v21, 0);
  if ( v30 )
    sub_B91220((__int64)&v30, v30);
  return 0;
}
