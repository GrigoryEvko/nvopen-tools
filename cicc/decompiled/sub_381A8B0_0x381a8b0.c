// Function: sub_381A8B0
// Address: 0x381a8b0
//
void __fastcall sub_381A8B0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rsi
  __int64 v6; // r12
  __int64 v7; // rsi
  __int64 v8; // rax
  __int128 v9; // xmm0
  __int128 v10; // xmm1
  unsigned __int16 *v11; // rax
  __int64 (__fastcall *v12)(__int64, __int64, __int64, __int64, __int64); // r13
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // rdx
  __int64 v16; // r13
  __int64 v17; // r9
  __int64 v18; // rax
  __int64 v19; // rdx
  unsigned int v20; // edx
  __int128 v21; // [rsp-10h] [rbp-B0h]
  __int64 v22; // [rsp+8h] [rbp-98h]
  __int64 v23; // [rsp+10h] [rbp-90h]
  __int64 v24; // [rsp+18h] [rbp-88h]
  __int64 v27; // [rsp+40h] [rbp-60h]
  unsigned __int8 *v28; // [rsp+50h] [rbp-50h]
  __int64 v29; // [rsp+60h] [rbp-40h] BYREF
  int v30; // [rsp+68h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 80);
  v29 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v29, v5, 1);
  v6 = *a1;
  v7 = a1[1];
  v30 = *(_DWORD *)(a2 + 72);
  v8 = *(_QWORD *)(a2 + 40);
  v9 = (__int128)_mm_loadu_si128((const __m128i *)v8);
  v10 = (__int128)_mm_loadu_si128((const __m128i *)(v8 + 40));
  v11 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v8 + 48LL) + 16LL * *(unsigned int *)(v8 + 8));
  v24 = *(_QWORD *)(v7 + 64);
  v22 = *((_QWORD *)v11 + 1);
  v12 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v6 + 528LL);
  v23 = *v11;
  v13 = sub_2E79000(*(__int64 **)(v7 + 40));
  v14 = v12(v6, v13, v24, v23, v22);
  v16 = v15;
  v18 = sub_340F900(
          (_QWORD *)a1[1],
          0xD0u,
          (__int64)&v29,
          v14,
          v15,
          v17,
          v9,
          v10,
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
  v27 = v19;
  *((_QWORD *)&v21 + 1) = v16;
  *(_QWORD *)&v21 = v14;
  v28 = sub_33FB620(
          a1[1],
          v18,
          v19,
          (__int64)&v29,
          **(unsigned __int16 **)(a2 + 48),
          *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
          (__m128i)v9,
          v21);
  sub_375BC20(a1, (__int64)v28, v20 | v27 & 0xFFFFFFFF00000000LL, a3, a4, (__m128i)v9);
  if ( v29 )
    sub_B91220((__int64)&v29, v29);
}
