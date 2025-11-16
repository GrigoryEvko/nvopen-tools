// Function: sub_2127DE0
// Address: 0x2127de0
//
__int64 *__fastcall sub_2127DE0(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v6; // r13
  unsigned int v7; // r12d
  __int64 v8; // rsi
  unsigned int v9; // r14d
  __int16 v10; // ax
  __int64 v11; // rax
  __int64 v12; // r9
  __int64 *v13; // r15
  __int64 v14; // r12
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // r13
  unsigned __int64 v18; // r8
  unsigned __int8 v19; // dl
  __int64 v20; // rcx
  __int128 v21; // rax
  __int64 *v22; // r12
  bool v24; // al
  char v25; // al
  unsigned __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // [rsp-8h] [rbp-98h]
  unsigned __int64 v30; // [rsp+8h] [rbp-88h]
  const void **v31; // [rsp+28h] [rbp-68h]
  __int64 v32; // [rsp+30h] [rbp-60h] BYREF
  int v33; // [rsp+38h] [rbp-58h]
  unsigned __int8 v34[8]; // [rsp+40h] [rbp-50h] BYREF
  unsigned __int64 v35; // [rsp+48h] [rbp-48h]
  const void **v36; // [rsp+50h] [rbp-40h]

  sub_1F40D10(
    (__int64)v34,
    *(_QWORD *)a1,
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = (unsigned __int8)v35;
  v7 = *(unsigned __int16 *)(a2 + 24);
  v8 = *(_QWORD *)(a2 + 72);
  v31 = v36;
  v9 = (unsigned __int8)v35;
  v32 = v8;
  v10 = v7;
  if ( v8 )
  {
    sub_1623A60((__int64)&v32, v8, 2);
    v10 = *(_WORD *)(a2 + 24);
  }
  v33 = *(_DWORD *)(a2 + 64);
  if ( v10 == 153 )
  {
    v27 = *(_QWORD *)a1;
    if ( (_BYTE)v6 == 1 )
    {
      v28 = 1;
      if ( !*(_BYTE *)(v27 + 2834) )
        goto LABEL_4;
    }
    else
    {
      if ( !(_BYTE)v6 )
        goto LABEL_4;
      v28 = (unsigned __int8)v6;
      if ( !*(_QWORD *)(v27 + 8 * v6 + 120) || !*(_BYTE *)(v27 + 259LL * (unsigned __int8)v6 + 2575) )
        goto LABEL_4;
    }
    if ( (*(_BYTE *)(v27 + 259 * v28 + 2574) & 0xFB) == 0 )
      v7 = 152;
  }
LABEL_4:
  v11 = sub_1D309E0(
          *(__int64 **)(a1 + 8),
          v7,
          (__int64)&v32,
          v9,
          v31,
          0,
          a3,
          a4,
          *(double *)a5.m128i_i64,
          *(_OWORD *)*(_QWORD *)(a2 + 32));
  v13 = *(__int64 **)(a1 + 8);
  v14 = v11;
  v15 = *(_QWORD *)(a2 + 40);
  v17 = v16;
  v18 = *(_QWORD *)(v15 + 8);
  v19 = *(_BYTE *)v15;
  v20 = v29;
  v34[0] = v19;
  v35 = v18;
  if ( v19 )
  {
    if ( (unsigned __int8)(v19 - 14) <= 0x5Fu )
    {
      switch ( v19 )
      {
        case 0x18u:
        case 0x19u:
        case 0x1Au:
        case 0x1Bu:
        case 0x1Cu:
        case 0x1Du:
        case 0x1Eu:
        case 0x1Fu:
        case 0x20u:
        case 0x3Eu:
        case 0x3Fu:
        case 0x40u:
        case 0x41u:
        case 0x42u:
        case 0x43u:
          v19 = 3;
          break;
        case 0x21u:
        case 0x22u:
        case 0x23u:
        case 0x24u:
        case 0x25u:
        case 0x26u:
        case 0x27u:
        case 0x28u:
        case 0x44u:
        case 0x45u:
        case 0x46u:
        case 0x47u:
        case 0x48u:
        case 0x49u:
          v19 = 4;
          break;
        case 0x29u:
        case 0x2Au:
        case 0x2Bu:
        case 0x2Cu:
        case 0x2Du:
        case 0x2Eu:
        case 0x2Fu:
        case 0x30u:
        case 0x4Au:
        case 0x4Bu:
        case 0x4Cu:
        case 0x4Du:
        case 0x4Eu:
        case 0x4Fu:
          v19 = 5;
          break;
        case 0x31u:
        case 0x32u:
        case 0x33u:
        case 0x34u:
        case 0x35u:
        case 0x36u:
        case 0x50u:
        case 0x51u:
        case 0x52u:
        case 0x53u:
        case 0x54u:
        case 0x55u:
          v19 = 6;
          break;
        case 0x37u:
          v19 = 7;
          break;
        case 0x56u:
        case 0x57u:
        case 0x58u:
        case 0x62u:
        case 0x63u:
        case 0x64u:
          v19 = 8;
          break;
        case 0x59u:
        case 0x5Au:
        case 0x5Bu:
        case 0x5Cu:
        case 0x5Du:
        case 0x65u:
        case 0x66u:
        case 0x67u:
        case 0x68u:
        case 0x69u:
          v19 = 9;
          break;
        case 0x5Eu:
        case 0x5Fu:
        case 0x60u:
        case 0x61u:
        case 0x6Au:
        case 0x6Bu:
        case 0x6Cu:
        case 0x6Du:
          v19 = 10;
          break;
        default:
          v19 = 2;
          break;
      }
      v18 = 0;
    }
  }
  else
  {
    v30 = v18;
    v24 = sub_1F58D20((__int64)v34);
    v19 = 0;
    v18 = v30;
    if ( v24 )
    {
      v25 = sub_1F596B0((__int64)v34);
      v18 = v26;
      v19 = v25;
    }
  }
  *(_QWORD *)&v21 = sub_1D2EF30(v13, v19, v18, v20, v18, v12);
  v22 = sub_1D332F0(
          v13,
          (unsigned int)(*(_WORD *)(a2 + 24) == 153) + 3,
          (__int64)&v32,
          v9,
          v31,
          0,
          a3,
          a4,
          a5,
          v14,
          v17,
          v21);
  if ( v32 )
    sub_161E7C0((__int64)&v32, v32);
  return v22;
}
