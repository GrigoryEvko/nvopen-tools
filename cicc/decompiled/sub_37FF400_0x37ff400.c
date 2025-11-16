// Function: sub_37FF400
// Address: 0x37ff400
//
void __fastcall sub_37FF400(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rsi
  __int128 v9; // xmm0
  __m128i v10; // xmm1
  __int64 v11; // r9
  __int64 (__fastcall *v12)(__int64, __int64, unsigned int, __int64); // r11
  __int16 *v13; // rax
  unsigned __int16 v14; // di
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // rsi
  int v18; // edx
  void *v19; // rax
  void *v20; // rbx
  int v21; // edx
  _QWORD *i; // r13
  __int64 v23; // rdx
  __int64 v24; // [rsp+0h] [rbp-C0h]
  void *v25; // [rsp+10h] [rbp-B0h]
  __m128i *v27; // [rsp+40h] [rbp-80h]
  __int64 v28; // [rsp+50h] [rbp-70h] BYREF
  int v29; // [rsp+58h] [rbp-68h]
  unsigned int v30; // [rsp+60h] [rbp-60h] BYREF
  __int64 v31; // [rsp+68h] [rbp-58h]
  void *v32; // [rsp+70h] [rbp-50h] BYREF
  _QWORD *v33; // [rsp+78h] [rbp-48h]
  __int64 v34; // [rsp+80h] [rbp-40h]

  if ( *(_DWORD *)(a2 + 24) != 298 || (*(_BYTE *)(a2 + 33) & 0xC) != 0 || (*(_WORD *)(a2 + 32) & 0x380) != 0 )
  {
    v7 = *(_QWORD *)(a2 + 40);
    v8 = *(_QWORD *)(a2 + 80);
    v9 = (__int128)_mm_loadu_si128((const __m128i *)v7);
    v10 = _mm_loadu_si128((const __m128i *)(v7 + 40));
    v28 = v8;
    if ( v8 )
      sub_B96E90((__int64)&v28, v8, 1);
    v11 = *(_QWORD *)a1;
    v29 = *(_DWORD *)(a2 + 72);
    v12 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v11 + 592LL);
    v13 = *(__int16 **)(a2 + 48);
    v14 = *v13;
    v15 = *((_QWORD *)v13 + 1);
    v16 = *(_QWORD *)(a1 + 8);
    if ( v12 == sub_2D56A50 )
    {
      sub_2FE6CC0((__int64)&v32, v11, *(_QWORD *)(v16 + 64), v14, v15);
      LOWORD(v30) = (_WORD)v33;
      v31 = v34;
    }
    else
    {
      v30 = v12(v11, *(_QWORD *)(v16 + 64), v14, v15);
      v31 = v23;
    }
    v17 = (*(_BYTE *)(a2 + 33) >> 2) & 3;
    v27 = sub_33F1B30(
            *(__int64 **)(a1 + 8),
            v17,
            (__int64)&v28,
            v30,
            v31,
            *(const __m128i **)(a2 + 112),
            v9,
            v10.m128i_i64[0],
            v10.m128i_i64[1],
            *(unsigned __int16 *)(a2 + 96),
            *(_QWORD *)(a2 + 104));
    *(_QWORD *)a4 = v27;
    *(_DWORD *)(a4 + 8) = v18;
    v24 = *(_QWORD *)(a1 + 8);
    v25 = sub_300AC80((unsigned __int16 *)&v30, v17);
    v19 = sub_C33340();
    v20 = v19;
    if ( v25 == v19 )
      sub_C3C500(&v32, (__int64)v19);
    else
      sub_C373C0(&v32, (__int64)v25);
    if ( v32 == v20 )
      sub_C3CEB0(&v32, 0);
    else
      sub_C37310((__int64)&v32, 0);
    *(_QWORD *)a3 = sub_33FE6E0(v24, (__int64 *)&v32, (__int64)&v28, v30, v31, 0, (__m128i)v9);
    *(_DWORD *)(a3 + 8) = v21;
    if ( v32 == v20 )
    {
      if ( v33 )
      {
        for ( i = &v33[3 * *(v33 - 1)]; v33 != i; sub_91D830(i) )
          i -= 3;
        j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
      }
    }
    else
    {
      sub_C338F0((__int64)&v32);
    }
    sub_3760E70(a1, a2, 1, (unsigned __int64)v27, *((_QWORD *)&v9 + 1) & 0xFFFFFFFF00000000LL | 1);
    if ( v28 )
      sub_B91220((__int64)&v28, v28);
  }
  else
  {
    sub_3846760();
  }
}
