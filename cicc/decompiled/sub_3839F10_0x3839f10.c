// Function: sub_3839F10
// Address: 0x3839f10
//
__m128i *__fastcall sub_3839F10(__int64 a1, __int64 a2)
{
  unsigned __int64 *v4; // rax
  __int64 v5; // rsi
  unsigned __int64 v6; // rcx
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // r13
  unsigned __int64 v9; // rsi
  char v10; // di
  __int64 v11; // r9
  int v12; // esi
  unsigned int v13; // r8d
  __int64 v14; // rax
  int v15; // r11d
  __m128i *v16; // r12
  __int64 v18; // rsi
  __int64 v19; // rax
  int v20; // eax
  int v21; // ecx
  int *v22; // [rsp+8h] [rbp-68h]
  unsigned __int64 v23; // [rsp+10h] [rbp-60h]
  unsigned __int64 v24; // [rsp+18h] [rbp-58h]
  int v25; // [rsp+2Ch] [rbp-44h] BYREF
  __int64 v26; // [rsp+30h] [rbp-40h] BYREF
  int v27; // [rsp+38h] [rbp-38h]

  v4 = *(unsigned __int64 **)(a2 + 40);
  v5 = *(_QWORD *)(a2 + 80);
  v6 = *v4;
  v7 = v4[10];
  v26 = v5;
  v8 = v4[11];
  v24 = v6;
  v23 = v4[1];
  if ( v5 )
  {
    sub_B96E90((__int64)&v26, v5, 1);
    v4 = *(unsigned __int64 **)(a2 + 40);
  }
  v9 = v4[5];
  v27 = *(_DWORD *)(a2 + 72);
  v25 = sub_375D5B0(a1, v9, v4[6]);
  v22 = sub_3805BC0(a1 + 712, &v25);
  sub_37593F0(a1, v22);
  v10 = *(_BYTE *)(a1 + 512) & 1;
  if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
  {
    v11 = a1 + 520;
    v12 = 7;
  }
  else
  {
    v18 = *(unsigned int *)(a1 + 528);
    v11 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v18 )
      goto LABEL_12;
    v12 = v18 - 1;
  }
  v13 = v12 & (37 * *v22);
  v14 = v11 + 24LL * v13;
  v15 = *(_DWORD *)v14;
  if ( *v22 == *(_DWORD *)v14 )
    goto LABEL_6;
  v20 = 1;
  while ( v15 != -1 )
  {
    v21 = v20 + 1;
    v13 = v12 & (v20 + v13);
    v14 = v11 + 24LL * v13;
    v15 = *(_DWORD *)v14;
    if ( *v22 == *(_DWORD *)v14 )
      goto LABEL_6;
    v20 = v21;
  }
  if ( v10 )
  {
    v19 = 192;
    goto LABEL_13;
  }
  v18 = *(unsigned int *)(a1 + 528);
LABEL_12:
  v19 = 24 * v18;
LABEL_13:
  v14 = v11 + v19;
LABEL_6:
  v16 = sub_33F49B0(
          *(_QWORD **)(a1 + 8),
          v24,
          v23,
          (__int64)&v26,
          *(_QWORD *)(v14 + 8),
          *(unsigned int *)(v14 + 16),
          v7,
          v8,
          *(unsigned __int16 *)(a2 + 96),
          *(_QWORD *)(a2 + 104),
          *(const __m128i **)(a2 + 112));
  if ( v26 )
    sub_B91220((__int64)&v26, v26);
  return v16;
}
