// Function: sub_37FEB80
// Address: 0x37feb80
//
void __fastcall sub_37FEB80(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4, __m128i a5)
{
  int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // r10
  __int64 v11; // r11
  __int64 v12; // rax
  __int64 v13; // rsi
  __int16 *v14; // rcx
  unsigned __int16 v15; // si
  __int64 (__fastcall *v16)(__int64, __int64, unsigned int); // r9
  int v17; // ecx
  __int64 v18; // [rsp+0h] [rbp-E0h]
  __int64 v19; // [rsp+8h] [rbp-D8h]
  char v20; // [rsp+17h] [rbp-C9h]
  _WORD *v21; // [rsp+18h] [rbp-C8h]
  __int64 v22; // [rsp+20h] [rbp-C0h] BYREF
  int v23; // [rsp+28h] [rbp-B8h]
  __int64 v24[4]; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v25; // [rsp+50h] [rbp-90h]
  __int64 v26; // [rsp+58h] [rbp-88h]
  __int64 v27; // [rsp+60h] [rbp-80h]
  __int64 v28; // [rsp+68h] [rbp-78h]
  __int64 v29; // [rsp+70h] [rbp-70h]
  __m128i v30; // [rsp+80h] [rbp-60h] BYREF
  __m128i v31; // [rsp+90h] [rbp-50h]
  __m128i v32; // [rsp+A0h] [rbp-40h]

  v8 = *(_DWORD *)(a2 + 24);
  v9 = *(_QWORD *)(a2 + 40);
  if ( v8 > 239 )
  {
    if ( (unsigned int)(v8 - 242) > 1 )
    {
LABEL_4:
      a5 = _mm_loadu_si128((const __m128i *)v9);
      v10 = 0;
      v20 = 0;
      v30 = a5;
      v31 = _mm_loadu_si128((const __m128i *)(v9 + 40));
      v11 = 0;
      v32 = _mm_loadu_si128((const __m128i *)(v9 + 80));
      goto LABEL_5;
    }
  }
  else if ( v8 <= 237 && (unsigned int)(v8 - 101) > 0x2F )
  {
    goto LABEL_4;
  }
  v20 = 1;
  v30 = _mm_loadu_si128((const __m128i *)(v9 + 40));
  v31 = _mm_loadu_si128((const __m128i *)(v9 + 80));
  v32 = _mm_loadu_si128((const __m128i *)(v9 + 120));
  v10 = *(_QWORD *)v9;
  v11 = *(_QWORD *)(v9 + 8);
LABEL_5:
  v12 = *a1;
  v13 = *(_QWORD *)(a2 + 80);
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  LOBYTE(v29) = 4;
  v21 = (_WORD *)v12;
  v22 = v13;
  if ( v13 )
  {
    v18 = v10;
    v19 = v11;
    sub_B96E90((__int64)&v22, v13, 1);
    v10 = v18;
    v11 = v19;
  }
  v14 = *(__int16 **)(a2 + 48);
  v23 = *(_DWORD *)(a2 + 72);
  v15 = *v14;
  v16 = (__int64 (__fastcall *)(__int64, __int64, unsigned int))*((_QWORD *)v14 + 1);
  v17 = 80;
  if ( v15 != 12 )
  {
    v17 = 81;
    if ( v15 != 13 )
    {
      v17 = 82;
      if ( v15 != 14 )
      {
        v17 = 83;
        if ( v15 != 15 )
        {
          v17 = 729;
          if ( v15 == 16 )
            v17 = 84;
        }
      }
    }
  }
  sub_3494590(
    (__int64)v24,
    v21,
    a1[1],
    v17,
    v15,
    v16,
    (__int64)&v30,
    3u,
    v25,
    v26,
    v27,
    v28,
    v29,
    (__int64)&v22,
    v10,
    v11);
  if ( v22 )
    sub_B91220((__int64)&v22, v22);
  if ( v20 )
    sub_3760E70((__int64)a1, a2, 1, v24[2], v24[3]);
  sub_375AEA0(a1, v24[0], v24[1], a3, a4, a5);
}
