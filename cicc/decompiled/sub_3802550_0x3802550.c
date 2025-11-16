// Function: sub_3802550
// Address: 0x3802550
//
unsigned __int64 __fastcall sub_3802550(_QWORD *a1, unsigned __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  unsigned __int16 v5; // cx
  __int64 v6; // r11
  int v7; // eax
  char v8; // r9
  __int64 v9; // r14
  char v10; // r10
  __int64 v11; // r15
  __int64 v12; // rax
  int v13; // eax
  __int64 v14; // r11
  _WORD *v15; // rsi
  unsigned __int64 v16; // r13
  __int64 v18; // rax
  unsigned __int16 v19; // [rsp+0h] [rbp-C0h]
  __int64 v20; // [rsp+8h] [rbp-B8h]
  char v21; // [rsp+8h] [rbp-B8h]
  __int64 v22; // [rsp+10h] [rbp-B0h] BYREF
  int v23; // [rsp+18h] [rbp-A8h]
  __m128i v24; // [rsp+20h] [rbp-A0h] BYREF
  unsigned int v25; // [rsp+30h] [rbp-90h] BYREF
  __int64 (__fastcall *v26)(__int64, __int64, unsigned int); // [rsp+38h] [rbp-88h]
  unsigned __int64 v27[8]; // [rsp+40h] [rbp-80h] BYREF
  __int64 v28; // [rsp+80h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 48);
  v4 = *(_QWORD *)(a2 + 80);
  v5 = *(_WORD *)v3;
  v6 = *(_QWORD *)(v3 + 8);
  v22 = v4;
  if ( v4 )
  {
    v19 = v5;
    v20 = v6;
    sub_B96E90((__int64)&v22, v4, 1);
    v5 = v19;
    v6 = v20;
  }
  v23 = *(_DWORD *)(a2 + 72);
  v7 = *(_DWORD *)(a2 + 24);
  if ( v7 > 239 )
  {
    if ( (unsigned int)(v7 - 242) > 1 )
      goto LABEL_15;
  }
  else
  {
    if ( v7 <= 237 && (unsigned int)(v7 - 101) > 0x2F )
    {
      if ( v7 == 226 )
      {
        v8 = 1;
LABEL_8:
        v9 = 0;
        v10 = 0;
        v24 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
        v11 = 0;
        goto LABEL_9;
      }
LABEL_15:
      v8 = v7 == 141;
      goto LABEL_8;
    }
    if ( v7 == 226 )
    {
      v8 = 1;
      goto LABEL_18;
    }
  }
  v8 = v7 == 141;
LABEL_18:
  v18 = *(_QWORD *)(a2 + 40);
  v10 = 1;
  v24 = _mm_loadu_si128((const __m128i *)(v18 + 40));
  v9 = *(_QWORD *)v18;
  v11 = *(_QWORD *)(v18 + 8);
LABEL_9:
  v26 = 0;
  LOWORD(v25) = 0;
  v21 = v10;
  v12 = *(_QWORD *)(v24.m128i_i64[0] + 48) + 16LL * v24.m128i_u32[2];
  v13 = sub_37FBE80(*(_WORD *)v12, *(_QWORD *)(v12 + 8), v5, v6, (__int64)&v25, v8);
  v14 = a1[1];
  v15 = (_WORD *)*a1;
  memset(&v27[5], 0, 24);
  LOBYTE(v28) = 4;
  sub_3494590((__int64)v27, v15, v14, v13, v25, v26, (__int64)&v24, 1u, 0, 0, 0, 0, 4, (__int64)&v22, v9, v11);
  if ( v21 )
  {
    sub_3760E70((__int64)a1, a2, 1, v27[2], v27[3]);
    v16 = 0;
    sub_3760E70((__int64)a1, a2, 0, v27[0], v27[1]);
  }
  else
  {
    v16 = v27[0];
  }
  if ( v22 )
    sub_B91220((__int64)&v22, v22);
  return v16;
}
