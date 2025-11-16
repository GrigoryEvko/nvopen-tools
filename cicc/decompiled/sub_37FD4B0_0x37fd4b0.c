// Function: sub_37FD4B0
// Address: 0x37fd4b0
//
unsigned __int8 *__fastcall sub_37FD4B0(__int64 *a1, unsigned __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // r9
  __int128 v6; // xmm0
  __int64 v7; // r14
  __int64 v8; // r15
  __int16 *v9; // rax
  __int64 v10; // r10
  unsigned __int16 v11; // si
  __int64 v12; // r8
  __int64 (__fastcall *v13)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v14; // r11
  unsigned int v15; // r10d
  __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 *v18; // rdi
  __int64 v19; // rcx
  _QWORD *v20; // rax
  unsigned __int8 *v21; // rax
  unsigned __int8 *v22; // r15
  __int64 v24; // rdx
  __int64 v25; // [rsp+0h] [rbp-70h]
  unsigned int v26; // [rsp+8h] [rbp-68h]
  __int64 v27; // [rsp+20h] [rbp-50h] BYREF
  int v28; // [rsp+28h] [rbp-48h]
  __int64 v29; // [rsp+30h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 40);
  v5 = *a1;
  v6 = (__int128)_mm_loadu_si128((const __m128i *)(v4 + 40));
  v7 = *(_QWORD *)v4;
  v8 = *(_QWORD *)(v4 + 8);
  v9 = *(__int16 **)(a2 + 48);
  v10 = *(_QWORD *)(a1[1] + 64);
  v11 = *v9;
  v12 = *((_QWORD *)v9 + 1);
  v13 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v13 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v27, v5, v10, v11, v12);
    v14 = v29;
    v15 = (unsigned __int16)v28;
  }
  else
  {
    v15 = v13(v5, v10, v11, v12);
    v14 = v24;
  }
  v16 = *(_QWORD *)(a2 + 80);
  v27 = v16;
  if ( v16 )
  {
    v25 = v14;
    v26 = v15;
    sub_B96E90((__int64)&v27, v16, 1);
    v14 = v25;
    v15 = v26;
  }
  v17 = *(_QWORD *)(a2 + 40);
  v18 = (__int64 *)a1[1];
  v28 = *(_DWORD *)(a2 + 72);
  v19 = *(_QWORD *)(*(_QWORD *)(v17 + 120) + 96LL);
  v20 = *(_QWORD **)(v19 + 24);
  if ( *(_DWORD *)(v19 + 32) > 0x40u )
    v20 = (_QWORD *)*v20;
  v21 = sub_3411830(v18, v15, v14, (__int64)&v27, v7, v8, v6, *(_OWORD *)(v17 + 80), (unsigned int)v20);
  v22 = v21;
  if ( (unsigned __int8 *)a2 != v21 )
    sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v21, 1);
  if ( v27 )
    sub_B91220((__int64)&v27, v27);
  return v22;
}
