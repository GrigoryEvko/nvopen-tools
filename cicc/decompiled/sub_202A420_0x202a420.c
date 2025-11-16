// Function: sub_202A420
// Address: 0x202a420
//
__int64 __fastcall sub_202A420(__int64 a1, __int64 a2, unsigned int a3, double a4, double a5, __m128i a6)
{
  unsigned int v6; // ecx
  unsigned __int8 *v8; // rax
  __int64 v9; // rsi
  unsigned int v10; // r15d
  const void **v11; // r14
  unsigned __int64 *v12; // rax
  unsigned __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rax
  char v16; // cl
  __int64 v17; // rax
  __int64 v18; // rsi
  __int128 v19; // rax
  __int64 v20; // r14
  __int64 v23; // [rsp+10h] [rbp-90h] BYREF
  unsigned __int64 v24; // [rsp+18h] [rbp-88h]
  __int128 v25; // [rsp+20h] [rbp-80h] BYREF
  __int64 v26; // [rsp+30h] [rbp-70h] BYREF
  int v27; // [rsp+38h] [rbp-68h]
  _QWORD v28[2]; // [rsp+40h] [rbp-60h] BYREF
  __int64 v29[10]; // [rsp+50h] [rbp-50h] BYREF

  v6 = a3;
  v8 = *(unsigned __int8 **)(a2 + 40);
  v9 = *(_QWORD *)(a2 + 72);
  v10 = *v8;
  v11 = (const void **)*((_QWORD *)v8 + 1);
  *(_QWORD *)&v25 = 0;
  v23 = 0;
  LODWORD(v24) = 0;
  DWORD2(v25) = 0;
  v26 = v9;
  if ( v9 )
  {
    sub_1623A60((__int64)&v26, v9, 2);
    v6 = a3;
  }
  v27 = *(_DWORD *)(a2 + 64);
  v12 = (unsigned __int64 *)(*(_QWORD *)(a2 + 32) + 40LL * v6);
  v13 = *v12;
  v14 = v12[1];
  v15 = *(_QWORD *)(*v12 + 40) + 16LL * *((unsigned int *)v12 + 2);
  v16 = *(_BYTE *)v15;
  v17 = *(_QWORD *)(v15 + 8);
  LOBYTE(v28[0]) = v16;
  v28[1] = v17;
  sub_2017DE0(a1, v13, v14, &v23, &v25);
  sub_1D19A30((__int64)v29, *(_QWORD *)(a1 + 8), v28);
  switch ( *(_WORD *)(a2 + 24) )
  {
    case 0xF6:
      v18 = 76;
      break;
    case 0xF7:
      v18 = 78;
      break;
    case 0xF8:
      v18 = 52;
      break;
    case 0xF9:
      v18 = 54;
      break;
    case 0xFA:
      v18 = 118;
      break;
    case 0xFB:
      v18 = 119;
      break;
    case 0xFC:
      v18 = 120;
      break;
    case 0xFD:
      v18 = 115;
      break;
    case 0xFE:
      v18 = 114;
      break;
    case 0xFF:
      v18 = 117;
      break;
    case 0x100:
      v18 = 116;
      break;
    case 0x101:
      v18 = (*(_BYTE *)(a2 + 80) & 0x10) == 0 ? 183 : 181;
      break;
    case 0x102:
      v18 = (*(_BYTE *)(a2 + 80) & 0x10) == 0 ? 182 : 180;
      break;
  }
  *(_QWORD *)&v19 = sub_1D332F0(
                      *(__int64 **)(a1 + 8),
                      v18,
                      (__int64)&v26,
                      v29[0],
                      (const void **)v29[1],
                      *(unsigned __int16 *)(a2 + 80),
                      a4,
                      a5,
                      a6,
                      v23,
                      v24,
                      v25);
  v20 = sub_1D309E0(
          *(__int64 **)(a1 + 8),
          *(unsigned __int16 *)(a2 + 24),
          (__int64)&v26,
          v10,
          v11,
          *(unsigned __int16 *)(a2 + 80),
          a4,
          a5,
          *(double *)a6.m128i_i64,
          v19);
  if ( v26 )
    sub_161E7C0((__int64)&v26, v26);
  return v20;
}
