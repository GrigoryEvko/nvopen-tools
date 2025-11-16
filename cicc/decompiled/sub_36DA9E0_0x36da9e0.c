// Function: sub_36DA9E0
// Address: 0x36da9e0
//
void __fastcall sub_36DA9E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rcx
  _QWORD *v11; // rax
  __m128i v12; // xmm2
  __m128i v13; // xmm1
  unsigned int v14; // ecx
  __m128i v15; // xmm0
  int v16; // eax
  int v17; // esi
  bool v18; // cf
  __int64 v19; // r8
  unsigned __int64 v20; // rcx
  _QWORD *v21; // rdi
  __int64 v22; // r13
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // [rsp+0h] [rbp-B0h] BYREF
  int v27; // [rsp+8h] [rbp-A8h]
  __m128i v28; // [rsp+10h] [rbp-A0h]
  __m128i v29; // [rsp+20h] [rbp-90h]
  _OWORD *v30; // [rsp+40h] [rbp-70h]
  __int64 v31; // [rsp+48h] [rbp-68h]
  _OWORD v32[6]; // [rsp+50h] [rbp-60h] BYREF

  v8 = *(_QWORD *)(a2 + 80);
  v26 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v26, v8, 1);
  v9 = *(_QWORD *)(a2 + 40);
  v27 = *(_DWORD *)(a2 + 72);
  v10 = *(_QWORD *)(*(_QWORD *)(v9 + 40) + 96LL);
  v11 = *(_QWORD **)(v10 + 24);
  if ( *(_DWORD *)(v10 + 32) > 0x40u )
    v11 = (_QWORD *)*v11;
  v12 = _mm_loadu_si128((const __m128i *)v9);
  v13 = _mm_loadu_si128((const __m128i *)(v9 + 80));
  v14 = (_DWORD)v11 - 8293;
  v15 = _mm_loadu_si128((const __m128i *)(v9 + 120));
  v16 = (_DWORD)v11 - 7895;
  v17 = 0;
  v18 = v14 < 5;
  v19 = *(unsigned int *)(a2 + 68);
  v20 = *(_QWORD *)(a2 + 48);
  if ( v18 )
    v17 = v16;
  v21 = *(_QWORD **)(a1 + 64);
  v28 = v13;
  v29 = v15;
  v32[2] = v12;
  v32[0] = v13;
  v32[1] = v15;
  v30 = v32;
  v31 = 0x300000003LL;
  v22 = sub_33E66D0(v21, v17, (__int64)&v26, v20, v19, a6, (unsigned __int64 *)v32, 3);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v22, v23, v24, v25);
  sub_3421DB0(v22);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v30 != v32 )
    _libc_free((unsigned __int64)v30);
  if ( v26 )
    sub_B91220((__int64)&v26, v26);
}
