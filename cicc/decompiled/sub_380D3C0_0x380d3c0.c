// Function: sub_380D3C0
// Address: 0x380d3c0
//
__int64 *__fastcall sub_380D3C0(__int64 a1, __int64 a2, __m128i a3)
{
  unsigned int v3; // r15d
  _QWORD *v6; // rax
  unsigned __int64 v7; // r13
  __int64 v8; // r12
  __int64 v9; // rsi
  __int64 v10; // r12
  unsigned int v11; // edx
  unsigned int v12; // r13d
  __int64 v13; // rax
  unsigned __int16 v14; // dx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rdx
  char v18; // al
  unsigned int v19; // eax
  int v20; // r9d
  __int64 v21; // rdi
  int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // r8
  __int16 v25; // ax
  __int64 v26; // rsi
  __int128 v27; // rax
  __int128 *v28; // rcx
  __int64 *v29; // rsi
  __int64 *v30; // r12
  __int64 v32; // rax
  __int64 v33; // [rsp+0h] [rbp-80h]
  __int64 v34; // [rsp+10h] [rbp-70h] BYREF
  int v35; // [rsp+18h] [rbp-68h]
  unsigned __int16 v36; // [rsp+20h] [rbp-60h] BYREF
  __int64 v37; // [rsp+28h] [rbp-58h]
  __int64 v38; // [rsp+30h] [rbp-50h] BYREF
  char v39; // [rsp+38h] [rbp-48h]
  __int64 v40; // [rsp+40h] [rbp-40h]
  __int64 v41; // [rsp+48h] [rbp-38h]

  v6 = *(_QWORD **)(a2 + 40);
  if ( *(_DWORD *)(a2 + 24) == 339 )
  {
    v7 = v6[5];
    v8 = v6[6];
  }
  else
  {
    v7 = v6[10];
    v8 = v6[11];
  }
  v9 = *(_QWORD *)(a2 + 80);
  v34 = v9;
  if ( v9 )
    sub_B96E90((__int64)&v34, v9, 1);
  v35 = *(_DWORD *)(a2 + 72);
  v10 = sub_380AAE0(a1, v7, v8);
  v12 = v11;
  v13 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 48LL)
      + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 48LL);
  v14 = *(_WORD *)v13;
  v15 = *(_QWORD *)(v13 + 8);
  v36 = v14;
  v37 = v15;
  if ( v14 )
  {
    if ( v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
      BUG();
    v32 = 16LL * (v14 - 1);
    v17 = *(_QWORD *)&byte_444C4A0[v32];
    v18 = byte_444C4A0[v32 + 8];
  }
  else
  {
    v40 = sub_3007260((__int64)&v36);
    v41 = v16;
    v17 = v40;
    v18 = v41;
  }
  v38 = v17;
  v39 = v18;
  v19 = sub_CA1930(&v38);
  v21 = *(_QWORD *)(a1 + 8);
  switch ( v19 )
  {
    case 1u:
      LOWORD(v22) = 2;
      break;
    case 2u:
      LOWORD(v22) = 3;
      break;
    case 4u:
      LOWORD(v22) = 4;
      break;
    case 8u:
      LOWORD(v22) = 5;
      break;
    case 0x10u:
      LOWORD(v22) = 6;
      break;
    case 0x20u:
      LOWORD(v22) = 7;
      break;
    case 0x40u:
      LOWORD(v22) = 8;
      break;
    case 0x80u:
      LOWORD(v22) = 9;
      break;
    default:
      v22 = sub_3007020(*(_QWORD **)(v21 + 64), v19);
      v21 = *(_QWORD *)(a1 + 8);
      HIWORD(v3) = HIWORD(v22);
      v24 = v23;
      goto LABEL_18;
  }
  v24 = 0;
LABEL_18:
  LOWORD(v3) = v22;
  v25 = *(_WORD *)(*(_QWORD *)(v10 + 48) + 16LL * v12);
  if ( v25 == 11 )
  {
    v26 = 236;
  }
  else if ( v36 == 11 )
  {
    v26 = 237;
  }
  else if ( v25 == 10 )
  {
    v26 = 240;
  }
  else
  {
    if ( v36 != 10 )
      sub_C64ED0("Attempt at an invalid promotion-related conversion", 1u);
    v26 = 241;
  }
  v33 = v24;
  *(_QWORD *)&v27 = sub_33FAF80(v21, v26, (__int64)&v34, v3, v24, v20, a3);
  v28 = *(__int128 **)(a2 + 40);
  if ( *(_DWORD *)(a2 + 24) == 339 )
    v29 = (__int64 *)(v28 + 5);
  else
    v29 = (__int64 *)v28 + 5;
  v30 = sub_33F34C0(
          *(__int64 **)(a1 + 8),
          339,
          (__int64)&v34,
          v3,
          v33,
          *(const __m128i **)(a2 + 112),
          *v28,
          v27,
          *v29,
          v29[1]);
  if ( v34 )
    sub_B91220((__int64)&v34, v34);
  return v30;
}
