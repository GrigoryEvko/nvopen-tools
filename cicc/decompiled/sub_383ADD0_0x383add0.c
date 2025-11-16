// Function: sub_383ADD0
// Address: 0x383add0
//
__int64 *__fastcall sub_383ADD0(__int64 a1, __int64 a2, __m128i a3)
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
  __int128 v26; // [rsp-10h] [rbp-80h]
  unsigned __int64 v27; // [rsp+8h] [rbp-68h]
  int *v28; // [rsp+8h] [rbp-68h]
  __int64 v29; // [rsp+10h] [rbp-60h]
  _QWORD *v30; // [rsp+18h] [rbp-58h]
  int v31; // [rsp+2Ch] [rbp-44h] BYREF
  __int64 v32; // [rsp+30h] [rbp-40h] BYREF
  int v33; // [rsp+38h] [rbp-38h]

  v30 = *(_QWORD **)(a1 + 8);
  v5 = *(_QWORD *)(a2 + 40);
  v6 = *(_QWORD *)(v5 + 80);
  v7 = *(_QWORD *)(v5 + 88);
  v8 = *(_QWORD *)(v6 + 80);
  v9 = (unsigned __int16 *)(*(_QWORD *)(v6 + 48) + 16LL * *(unsigned int *)(v5 + 88));
  v10 = *((_QWORD *)v9 + 1);
  v11 = *v9;
  v32 = v8;
  v29 = v10;
  if ( v8 )
  {
    v27 = v6;
    sub_B96E90((__int64)&v32, v8, 1);
    v6 = v27;
  }
  v33 = *(_DWORD *)(v6 + 72);
  v31 = sub_375D5B0(a1, v6, v7);
  v28 = sub_3805BC0(a1 + 712, &v31);
  sub_37593F0(a1, v28);
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
  v15 = v14 & (37 * *v28);
  v16 = v13 + 24LL * v15;
  v17 = *(_DWORD *)v16;
  if ( *v28 == *(_DWORD *)v16 )
    goto LABEL_6;
  v24 = 1;
  while ( v17 != -1 )
  {
    v25 = v24 + 1;
    v15 = v14 & (v24 + v15);
    v16 = v13 + 24LL * v15;
    v17 = *(_DWORD *)v16;
    if ( *v28 == *(_DWORD *)v16 )
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
          (__int64)&v32,
          v11,
          v29,
          a3);
  v20 = v19;
  if ( v32 )
    sub_B91220((__int64)&v32, v32);
  *((_QWORD *)&v26 + 1) = v20;
  *(_QWORD *)&v26 = v18;
  return sub_33EC3B0(
           v30,
           (__int64 *)a2,
           **(_QWORD **)(a2 + 40),
           *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
           *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
           *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
           v26);
}
