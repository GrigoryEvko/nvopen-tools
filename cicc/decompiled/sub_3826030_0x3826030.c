// Function: sub_3826030
// Address: 0x3826030
//
void __fastcall sub_3826030(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r9
  __int16 *v6; // rax
  __int64 v7; // rsi
  unsigned __int16 v8; // r8
  __int64 (__fastcall *v9)(__int64, __int64, unsigned int); // r15
  unsigned int v10; // r14d
  _WORD *v11; // r11
  __int64 *v12; // r10
  __int64 v13; // rax
  __m128i v14; // xmm0
  __m128i v15; // xmm1
  int v16; // ecx
  __int64 v17; // rsi
  unsigned int *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r9
  unsigned __int8 *v21; // rax
  __int128 v22; // [rsp-10h] [rbp-E0h]
  unsigned __int16 v23; // [rsp+0h] [rbp-D0h]
  __int64 v24; // [rsp+10h] [rbp-C0h]
  __int64 *v25; // [rsp+10h] [rbp-C0h]
  __int64 v27; // [rsp+20h] [rbp-B0h] BYREF
  int v28; // [rsp+28h] [rbp-A8h]
  _OWORD v29[2]; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v30[8]; // [rsp+50h] [rbp-80h] BYREF
  __int64 v31; // [rsp+90h] [rbp-40h]

  v4 = a2;
  v6 = *(__int16 **)(a2 + 48);
  v7 = *(_QWORD *)(a2 + 80);
  v8 = *v6;
  v9 = (__int64 (__fastcall *)(__int64, __int64, unsigned int))*((_QWORD *)v6 + 1);
  v27 = v7;
  v10 = v8;
  if ( v7 )
  {
    v24 = v4;
    v23 = v8;
    sub_B96E90((__int64)&v27, v7, 1);
    v4 = v24;
    v8 = v23;
  }
  v11 = (_WORD *)*a1;
  v12 = (__int64 *)a1[1];
  v28 = *(_DWORD *)(v4 + 72);
  v13 = *(_QWORD *)(v4 + 40);
  v14 = _mm_loadu_si128((const __m128i *)v13);
  v15 = _mm_loadu_si128((const __m128i *)(v13 + 40));
  v29[0] = v14;
  v29[1] = v15;
  if ( !v8 )
  {
    v16 = 729;
    goto LABEL_6;
  }
  if ( HIBYTE(v11[250 * v8 + 3239]) != 4 )
  {
    v16 = 31;
    if ( v8 != 6 )
    {
      v16 = 32;
      if ( v8 != 7 )
      {
        v16 = 33;
        if ( v8 != 8 )
        {
          v16 = 34;
          if ( v8 != 9 )
            v16 = 729;
        }
      }
    }
LABEL_6:
    memset(&v30[5], 0, 24);
    LOBYTE(v31) = 5;
    sub_3494590((__int64)v30, v11, (__int64)v12, v16, v10, v9, (__int64)v29, 2u, 0, 0, 0, 0, 5, (__int64)&v27, 0, 0);
    sub_375BC20(a1, v30[0], v30[1], a3, a4, v14);
    v17 = v27;
    if ( !v27 )
      return;
    goto LABEL_7;
  }
  v25 = v12;
  v18 = (unsigned int *)sub_33E5110(v12, v10, (__int64)v9, v10, (__int64)v9);
  *((_QWORD *)&v22 + 1) = 2;
  *(_QWORD *)&v22 = v29;
  v21 = sub_3411630(v25, 65, (__int64)&v27, v18, v19, v20, v22);
  sub_375BC20(a1, (__int64)v21, 1, a3, a4, v14);
  v17 = v27;
  if ( !v27 )
    return;
LABEL_7:
  sub_B91220((__int64)&v27, v17);
}
