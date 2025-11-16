// Function: sub_37FD230
// Address: 0x37fd230
//
__int64 *__fastcall sub_37FD230(__int64 *a1, unsigned __int64 a2)
{
  __int16 *v4; // rax
  unsigned __int16 v5; // si
  __int64 v6; // r8
  __int64 (__fastcall *v7)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v8; // rbx
  unsigned int v9; // r10d
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 *v12; // rdi
  bool v13; // zf
  const __m128i *v14; // r9
  unsigned __int64 v15; // rax
  __int64 v16; // rdx
  __int64 *v17; // r14
  __int64 v19; // rdx
  const __m128i *v20; // [rsp+8h] [rbp-78h]
  unsigned __int16 v21; // [rsp+10h] [rbp-70h]
  unsigned int v22; // [rsp+18h] [rbp-68h]
  __int64 v23; // [rsp+20h] [rbp-60h] BYREF
  int v24; // [rsp+28h] [rbp-58h]
  __m128i v25; // [rsp+30h] [rbp-50h] BYREF
  __m128i v26; // [rsp+40h] [rbp-40h]

  v4 = *(__int16 **)(a2 + 48);
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  v7 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v7 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v25, *a1, *(_QWORD *)(a1[1] + 64), v5, v6);
    v8 = v26.m128i_i64[0];
    v9 = v25.m128i_u16[4];
  }
  else
  {
    v9 = v7(*a1, *(_QWORD *)(a1[1] + 64), v5, v6);
    v8 = v19;
  }
  v10 = *(_QWORD *)(a2 + 80);
  v23 = v10;
  if ( v10 )
  {
    v22 = v9;
    sub_B96E90((__int64)&v23, v10, 1);
    v9 = v22;
  }
  v24 = *(_DWORD *)(a2 + 72);
  if ( (*(_BYTE *)(a2 + 33) & 0xC) != 0 )
    sub_C64ED0("softening fp extending atomic load not handled", 1u);
  v11 = *(_QWORD *)(a2 + 40);
  v12 = (__int64 *)a1[1];
  v13 = *(_DWORD *)(a2 + 24) == 339;
  v14 = *(const __m128i **)(a2 + 112);
  v25 = _mm_loadu_si128((const __m128i *)v11);
  if ( v13 )
    v26 = _mm_loadu_si128((const __m128i *)(v11 + 80));
  else
    v26 = _mm_loadu_si128((const __m128i *)(v11 + 40));
  v20 = v14;
  v21 = v9;
  v15 = sub_33E5110(v12, v9, v8, 1, 0);
  v17 = sub_33E6BC0(v12, 338, (__int64)&v23, v21, v8, v20, v15, v16, (unsigned __int64 *)&v25, 2);
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v17, 1);
  if ( v23 )
    sub_B91220((__int64)&v23, v23);
  return v17;
}
