// Function: sub_3838540
// Address: 0x3838540
//
unsigned __int8 *__fastcall sub_3838540(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __m128i a6,
        __int64 a7,
        __int128 a8)
{
  unsigned __int64 v8; // r10
  __int64 v11; // rax
  __int64 v12; // rsi
  unsigned __int16 v13; // r12
  __int64 v14; // r13
  char v15; // si
  __int64 v16; // r10
  int v17; // ecx
  unsigned int v18; // edi
  __int64 v19; // rax
  int v20; // r11d
  unsigned __int8 *v21; // r12
  __int64 v23; // rcx
  __int64 v24; // rax
  int v25; // eax
  int v26; // r9d
  __int64 v27; // [rsp+8h] [rbp-68h]
  __int64 v28; // [rsp+8h] [rbp-68h]
  unsigned __int64 v29; // [rsp+10h] [rbp-60h]
  int *v30; // [rsp+10h] [rbp-60h]
  int v32; // [rsp+2Ch] [rbp-44h] BYREF
  __int64 v33; // [rsp+30h] [rbp-40h] BYREF
  int v34; // [rsp+38h] [rbp-38h]

  v8 = a2;
  v11 = *(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3;
  v12 = *(_QWORD *)(a2 + 80);
  v13 = *(_WORD *)v11;
  v14 = *(_QWORD *)(v11 + 8);
  v33 = v12;
  if ( v12 )
  {
    v27 = a5;
    v29 = v8;
    sub_B96E90((__int64)&v33, v12, 1);
    a5 = v27;
    v8 = v29;
  }
  v28 = a5;
  v34 = *(_DWORD *)(v8 + 72);
  v32 = sub_375D5B0(a1, v8, a3);
  v30 = sub_3805BC0(a1 + 712, &v32);
  sub_37593F0(a1, v30);
  v15 = *(_BYTE *)(a1 + 512) & 1;
  if ( v15 )
  {
    v16 = a1 + 520;
    v17 = 7;
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 528);
    v16 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v23 )
      goto LABEL_12;
    v17 = v23 - 1;
  }
  v18 = v17 & (37 * *v30);
  v19 = v16 + 24LL * v18;
  v20 = *(_DWORD *)v19;
  if ( *v30 == *(_DWORD *)v19 )
    goto LABEL_6;
  v25 = 1;
  while ( v20 != -1 )
  {
    v26 = v25 + 1;
    v18 = v17 & (v25 + v18);
    v19 = v16 + 24LL * v18;
    v20 = *(_DWORD *)v19;
    if ( *v30 == *(_DWORD *)v19 )
      goto LABEL_6;
    v25 = v26;
  }
  if ( v15 )
  {
    v24 = 192;
    goto LABEL_13;
  }
  v23 = *(unsigned int *)(a1 + 528);
LABEL_12:
  v24 = 24 * v23;
LABEL_13:
  v19 = v16 + v24;
LABEL_6:
  v21 = sub_3400810(
          *(_QWORD **)(a1 + 8),
          *(_QWORD *)(v19 + 8),
          *(unsigned int *)(v19 + 16) | a3 & 0xFFFFFFFF00000000LL,
          a4,
          v28,
          (__int64)&v33,
          a6,
          a8,
          v13,
          v14);
  if ( v33 )
    sub_B91220((__int64)&v33, v33);
  return v21;
}
