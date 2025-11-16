// Function: sub_375A6A0
// Address: 0x375a6a0
//
unsigned __int8 *__fastcall sub_375A6A0(__int64 a1, __int64 a2, unsigned int a3, __m128i a4)
{
  unsigned int v4; // r15d
  __int64 v6; // rax
  unsigned __int16 v7; // dx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  char v11; // al
  unsigned int v12; // eax
  __int64 v13; // r9
  int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // r8
  __int64 v17; // rsi
  unsigned __int8 *v18; // r12
  __int64 v20; // rax
  __int64 v21; // [rsp+0h] [rbp-60h]
  __int64 v22; // [rsp+8h] [rbp-58h]
  __int64 v23; // [rsp+8h] [rbp-58h]
  __int64 v24; // [rsp+10h] [rbp-50h] BYREF
  __int64 v25; // [rsp+18h] [rbp-48h]
  __int64 v26; // [rsp+20h] [rbp-40h]
  __int64 v27; // [rsp+28h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 48) + 16LL * a3;
  v7 = *(_WORD *)v6;
  v8 = *(_QWORD *)(v6 + 8);
  LOWORD(v24) = v7;
  v25 = v8;
  if ( v7 )
  {
    if ( v7 == 1 || (unsigned __int16)(v7 - 504) <= 7u )
      BUG();
    v20 = 16LL * (v7 - 1);
    v10 = *(_QWORD *)&byte_444C4A0[v20];
    v11 = byte_444C4A0[v20 + 8];
  }
  else
  {
    v26 = sub_3007260((__int64)&v24);
    v27 = v9;
    v10 = v26;
    v11 = v27;
  }
  v24 = v10;
  LOBYTE(v25) = v11;
  v12 = sub_CA1930(&v24);
  v13 = *(_QWORD *)(a1 + 8);
  switch ( v12 )
  {
    case 1u:
      LOWORD(v14) = 2;
      break;
    case 2u:
      LOWORD(v14) = 3;
      break;
    case 4u:
      LOWORD(v14) = 4;
      break;
    case 8u:
      LOWORD(v14) = 5;
      break;
    case 0x10u:
      LOWORD(v14) = 6;
      break;
    case 0x20u:
      LOWORD(v14) = 7;
      break;
    case 0x40u:
      LOWORD(v14) = 8;
      break;
    case 0x80u:
      LOWORD(v14) = 9;
      break;
    default:
      v22 = *(_QWORD *)(a1 + 8);
      v14 = sub_3007020(*(_QWORD **)(v13 + 64), v12);
      v13 = v22;
      HIWORD(v4) = HIWORD(v14);
      v16 = v15;
      goto LABEL_14;
  }
  v16 = 0;
LABEL_14:
  v17 = *(_QWORD *)(a2 + 80);
  LOWORD(v4) = v14;
  v24 = v17;
  if ( v17 )
  {
    v21 = v16;
    v23 = v13;
    sub_B96E90((__int64)&v24, v17, 1);
    v16 = v21;
    v13 = v23;
  }
  LODWORD(v25) = *(_DWORD *)(a2 + 72);
  v18 = sub_33FAF80(v13, 234, (__int64)&v24, v4, v16, v13, a4);
  if ( v24 )
    sub_B91220((__int64)&v24, v24);
  return v18;
}
