// Function: sub_383A840
// Address: 0x383a840
//
__int64 *__fastcall sub_383A840(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v5; // rax
  unsigned __int64 v6; // r10
  __int64 v7; // r14
  __int64 v8; // rsi
  unsigned __int16 *v9; // rax
  __int64 v10; // rcx
  unsigned int v11; // r15d
  char v12; // si
  __int64 v13; // r10
  int v14; // ecx
  unsigned int v15; // edi
  __int64 v16; // rax
  int v17; // r11d
  unsigned __int8 *v18; // r14
  __int64 v19; // rdx
  __int64 v20; // r15
  __int64 v22; // rcx
  __int64 v23; // rax
  int v24; // eax
  int v25; // r8d
  unsigned __int64 v26; // [rsp+8h] [rbp-68h]
  int *v27; // [rsp+8h] [rbp-68h]
  __int64 v28; // [rsp+10h] [rbp-60h]
  _QWORD *v29; // [rsp+18h] [rbp-58h]
  int v30; // [rsp+2Ch] [rbp-44h] BYREF
  __int64 v31; // [rsp+30h] [rbp-40h] BYREF
  int v32; // [rsp+38h] [rbp-38h]

  v29 = *(_QWORD **)(a1 + 8);
  v5 = *(_QWORD *)(a2 + 40);
  v6 = *(_QWORD *)(v5 + 40);
  v7 = *(_QWORD *)(v5 + 48);
  v8 = *(_QWORD *)(v6 + 80);
  v9 = (unsigned __int16 *)(*(_QWORD *)(v6 + 48) + 16LL * *(unsigned int *)(v5 + 48));
  v10 = *((_QWORD *)v9 + 1);
  v11 = *v9;
  v31 = v8;
  v28 = v10;
  if ( v8 )
  {
    v26 = v6;
    sub_B96E90((__int64)&v31, v8, 1);
    v6 = v26;
  }
  v32 = *(_DWORD *)(v6 + 72);
  v30 = sub_375D5B0(a1, v6, v7);
  v27 = sub_3805BC0(a1 + 712, &v30);
  sub_37593F0(a1, v27);
  v12 = *(_BYTE *)(a1 + 512) & 1;
  if ( v12 )
  {
    v13 = a1 + 520;
    v14 = 7;
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 528);
    v13 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v22 )
      goto LABEL_12;
    v14 = v22 - 1;
  }
  v15 = v14 & (37 * *v27);
  v16 = v13 + 24LL * v15;
  v17 = *(_DWORD *)v16;
  if ( *v27 == *(_DWORD *)v16 )
    goto LABEL_6;
  v24 = 1;
  while ( v17 != -1 )
  {
    v25 = v24 + 1;
    v15 = v14 & (v24 + v15);
    v16 = v13 + 24LL * v15;
    v17 = *(_DWORD *)v16;
    if ( *v27 == *(_DWORD *)v16 )
      goto LABEL_6;
    v24 = v25;
  }
  if ( v12 )
  {
    v23 = 192;
    goto LABEL_13;
  }
  v22 = *(unsigned int *)(a1 + 528);
LABEL_12:
  v23 = 24 * v22;
LABEL_13:
  v16 = v13 + v23;
LABEL_6:
  v18 = sub_34070B0(
          *(_QWORD **)(a1 + 8),
          *(_QWORD *)(v16 + 8),
          *(unsigned int *)(v16 + 16) | v7 & 0xFFFFFFFF00000000LL,
          (__int64)&v31,
          v11,
          v28,
          a3);
  v20 = v19;
  if ( v31 )
    sub_B91220((__int64)&v31, v31);
  return sub_33EC010(
           v29,
           (__int64 *)a2,
           **(_QWORD **)(a2 + 40),
           *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
           (__int64)v18,
           v20);
}
