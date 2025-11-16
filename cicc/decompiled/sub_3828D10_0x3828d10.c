// Function: sub_3828D10
// Address: 0x3828d10
//
unsigned __int8 *__fastcall sub_3828D10(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v5; // rbx
  unsigned __int64 v6; // r11
  __int64 v7; // rcx
  unsigned __int64 v8; // r15
  __int64 v9; // rbx
  unsigned __int16 *v10; // rax
  unsigned int *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r9
  unsigned __int8 *v14; // rax
  __int64 v15; // r9
  unsigned __int8 *v16; // r14
  __int128 v18; // [rsp-20h] [rbp-E0h]
  __int64 v19; // [rsp+0h] [rbp-C0h]
  unsigned __int64 v20; // [rsp+8h] [rbp-B8h]
  __int128 v21; // [rsp+10h] [rbp-B0h]
  __int128 v22; // [rsp+20h] [rbp-A0h]
  __int64 v23; // [rsp+30h] [rbp-90h]
  __int64 v24; // [rsp+38h] [rbp-88h]
  __int64 v25; // [rsp+40h] [rbp-80h] BYREF
  int v26; // [rsp+48h] [rbp-78h]
  __int128 v27; // [rsp+50h] [rbp-70h] BYREF
  __int128 v28; // [rsp+60h] [rbp-60h] BYREF
  __int128 v29; // [rsp+70h] [rbp-50h] BYREF
  __int128 v30; // [rsp+80h] [rbp-40h] BYREF

  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(a2 + 80);
  v5 = *(_QWORD *)(v3 + 48);
  v6 = *(_QWORD *)v3;
  v25 = v4;
  v7 = *(_QWORD *)(v3 + 8);
  v23 = v5;
  v8 = *(_QWORD *)(v3 + 40);
  v22 = (__int128)_mm_loadu_si128((const __m128i *)(v3 + 80));
  v21 = (__int128)_mm_loadu_si128((const __m128i *)(v3 + 120));
  v24 = *(_QWORD *)(v3 + 80);
  v9 = *(unsigned int *)(v3 + 88);
  if ( v4 )
  {
    v19 = *(_QWORD *)(v3 + 8);
    v20 = v6;
    sub_B96E90((__int64)&v25, v4, 1);
    v7 = v19;
    v6 = v20;
  }
  v26 = *(_DWORD *)(a2 + 72);
  *(_QWORD *)&v27 = 0;
  DWORD2(v27) = 0;
  *(_QWORD *)&v28 = 0;
  DWORD2(v28) = 0;
  *(_QWORD *)&v29 = 0;
  DWORD2(v29) = 0;
  *(_QWORD *)&v30 = 0;
  DWORD2(v30) = 0;
  sub_375E510(a1, v6, v7, (__int64)&v27, (__int64)&v28);
  sub_375E510(a1, v8, v23, (__int64)&v29, (__int64)&v30);
  v10 = (unsigned __int16 *)(*(_QWORD *)(v27 + 48) + 16LL * DWORD2(v27));
  v11 = (unsigned int *)sub_33E5110(
                          *(__int64 **)(a1 + 8),
                          *v10,
                          *((_QWORD *)v10 + 1),
                          *(unsigned __int16 *)(*(_QWORD *)(v24 + 48) + 16 * v9),
                          *(_QWORD *)(*(_QWORD *)(v24 + 48) + 16 * v9 + 8));
  v14 = sub_3412970(*(_QWORD **)(a1 + 8), 73, (__int64)&v25, v11, v12, v13, v27, v29, v22);
  *((_QWORD *)&v18 + 1) = 1;
  *(_QWORD *)&v18 = v14;
  v16 = sub_33FC130(
          *(_QWORD **)(a1 + 8),
          209,
          (__int64)&v25,
          **(unsigned __int16 **)(a2 + 48),
          *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
          v15,
          v28,
          v30,
          v18,
          v21);
  if ( v25 )
    sub_B91220((__int64)&v25, v25);
  return v16;
}
