// Function: sub_380C690
// Address: 0x380c690
//
unsigned __int8 *__fastcall sub_380C690(__int64 a1, __int64 a2, __m128i a3)
{
  unsigned int v3; // r15d
  __int64 v5; // rax
  unsigned __int16 *v6; // rdx
  unsigned __int16 v7; // cx
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rdx
  char v11; // al
  unsigned int v12; // eax
  int v13; // r9d
  __int64 v14; // r10
  int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // r8
  __int64 v18; // rsi
  __int64 v19; // rsi
  unsigned __int8 *v20; // r13
  unsigned int v21; // edx
  unsigned int v22; // r12d
  __int64 v24; // rax
  __int64 v25; // [rsp+0h] [rbp-80h]
  __int64 v26; // [rsp+8h] [rbp-78h]
  __int16 v27; // [rsp+16h] [rbp-6Ah]
  unsigned __int16 v28; // [rsp+20h] [rbp-60h] BYREF
  __int64 v29; // [rsp+28h] [rbp-58h]
  __int64 v30; // [rsp+30h] [rbp-50h] BYREF
  int v31; // [rsp+38h] [rbp-48h]
  __int64 v32; // [rsp+40h] [rbp-40h]
  __int64 v33; // [rsp+48h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 40);
  v6 = *(unsigned __int16 **)(*(_QWORD *)v5 + 48LL);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  v28 = v7;
  v29 = v8;
  v27 = **(_WORD **)(sub_380AAE0(a1, *(_QWORD *)v5, *(_QWORD *)(v5 + 8)) + 48);
  if ( v28 )
  {
    if ( v28 == 1 || (unsigned __int16)(v28 - 504) <= 7u )
      BUG();
    v24 = 16LL * (v28 - 1);
    v10 = *(_QWORD *)&byte_444C4A0[v24];
    v11 = byte_444C4A0[v24 + 8];
  }
  else
  {
    v32 = sub_3007260((__int64)&v28);
    v33 = v9;
    v10 = v32;
    v11 = v33;
  }
  v30 = v10;
  LOBYTE(v31) = v11;
  v12 = sub_CA1930(&v30);
  v14 = *(_QWORD *)(a1 + 8);
  switch ( v12 )
  {
    case 1u:
      LOWORD(v15) = 2;
      break;
    case 2u:
      LOWORD(v15) = 3;
      break;
    case 4u:
      LOWORD(v15) = 4;
      break;
    case 8u:
      LOWORD(v15) = 5;
      break;
    case 0x10u:
      LOWORD(v15) = 6;
      break;
    case 0x20u:
      LOWORD(v15) = 7;
      break;
    case 0x40u:
      LOWORD(v15) = 8;
      break;
    case 0x80u:
      LOWORD(v15) = 9;
      break;
    default:
      v15 = sub_3007020(*(_QWORD **)(v14 + 64), v12);
      HIWORD(v3) = HIWORD(v15);
      v17 = v16;
      v14 = *(_QWORD *)(a1 + 8);
      goto LABEL_14;
  }
  v17 = 0;
LABEL_14:
  v18 = *(_QWORD *)(a2 + 80);
  LOWORD(v3) = v15;
  v30 = v18;
  if ( v18 )
  {
    v25 = v17;
    v26 = v14;
    sub_B96E90((__int64)&v30, v18, 1);
    v17 = v25;
    v14 = v26;
  }
  v31 = *(_DWORD *)(a2 + 72);
  if ( v27 == 11 )
  {
    v19 = 236;
  }
  else if ( v28 == 11 )
  {
    v19 = 237;
  }
  else if ( v27 == 10 )
  {
    v19 = 240;
  }
  else
  {
    if ( v28 != 10 )
      sub_C64ED0("Attempt at an invalid promotion-related conversion", 1u);
    v19 = 241;
  }
  v20 = sub_33FAF80(v14, v19, (__int64)&v30, v3, v17, v13, a3);
  v22 = v21;
  if ( v30 )
    sub_B91220((__int64)&v30, v30);
  return sub_33FB890(
           *(_QWORD *)(a1 + 8),
           **(unsigned __int16 **)(a2 + 48),
           *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
           (__int64)v20,
           v22,
           a3);
}
