// Function: sub_213EEC0
// Address: 0x213eec0
//
__int64 __fastcall sub_213EEC0(__int64 **a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v6; // rsi
  __int64 v7; // r12
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // r13
  __int64 v10; // rax
  int v11; // eax
  __int64 *v12; // r10
  unsigned int v13; // eax
  unsigned __int8 v14; // r8
  __int64 v15; // rax
  __int64 *v16; // r10
  __int64 v17; // r8
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r9
  const void **v21; // rsi
  unsigned __int8 v22; // dl
  bool v23; // al
  __int64 *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r12
  unsigned __int8 v28; // al
  char v29; // al
  const void **v30; // rdx
  __int128 v31; // [rsp-10h] [rbp-A0h]
  __int64 v32; // [rsp+0h] [rbp-90h]
  __int64 v33; // [rsp+8h] [rbp-88h]
  __int64 (__fastcall *v34)(__int64, __int64); // [rsp+20h] [rbp-70h]
  __int64 *v35; // [rsp+20h] [rbp-70h]
  __int64 *v36; // [rsp+30h] [rbp-60h]
  __int64 *v37; // [rsp+38h] [rbp-58h]
  __int64 v38; // [rsp+40h] [rbp-50h] BYREF
  int v39; // [rsp+48h] [rbp-48h]
  unsigned __int8 v40[8]; // [rsp+50h] [rbp-40h] BYREF
  const void **v41; // [rsp+58h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 72);
  v38 = v6;
  if ( v6 )
    sub_1623A60((__int64)&v38, v6, 2);
  v39 = *(_DWORD *)(a2 + 64);
  v7 = sub_2138AD0((__int64)a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v9 = v8;
  v36 = *a1;
  v37 = a1[1];
  v34 = *(__int64 (__fastcall **)(__int64, __int64))(**a1 + 48);
  v10 = sub_1E0A0C0(v37[4]);
  if ( v34 == sub_1D13A20 )
  {
    v11 = sub_15A9520(v10, 0);
    v12 = v37;
    v13 = 8 * v11;
    if ( v13 == 32 )
    {
      v14 = 5;
    }
    else if ( v13 > 0x20 )
    {
      v14 = 6;
      if ( v13 != 64 )
      {
        v14 = 0;
        if ( v13 == 128 )
          v14 = 7;
      }
    }
    else
    {
      v14 = 3;
      if ( v13 != 8 )
        v14 = 4 * (v13 == 16);
    }
  }
  else
  {
    v28 = v34((__int64)v36, v10);
    v12 = v37;
    v14 = v28;
  }
  v15 = sub_1D323C0(
          v12,
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL),
          (__int64)&v38,
          v14,
          0,
          a3,
          a4,
          *(double *)a5.m128i_i64);
  v16 = a1[1];
  v17 = v15;
  v18 = *(_QWORD *)(v7 + 40);
  v20 = v19;
  v21 = *(const void ***)(v18 + 8);
  v22 = *(_BYTE *)v18;
  v41 = v21;
  v40[0] = v22;
  if ( v22 )
  {
    if ( (unsigned __int8)(v22 - 14) <= 0x5Fu )
    {
      switch ( v22 )
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
          v22 = 3;
          v21 = 0;
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
          v22 = 4;
          v21 = 0;
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
          v22 = 5;
          v21 = 0;
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
          v22 = 6;
          v21 = 0;
          break;
        case 0x37u:
          v22 = 7;
          v21 = 0;
          break;
        case 0x56u:
        case 0x57u:
        case 0x58u:
        case 0x62u:
        case 0x63u:
        case 0x64u:
          v22 = 8;
          v21 = 0;
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
          v22 = 9;
          v21 = 0;
          break;
        case 0x5Eu:
        case 0x5Fu:
        case 0x60u:
        case 0x61u:
        case 0x6Au:
        case 0x6Bu:
        case 0x6Cu:
        case 0x6Du:
          v22 = 10;
          v21 = 0;
          break;
        default:
          v22 = 2;
          v21 = 0;
          break;
      }
    }
  }
  else
  {
    v32 = v17;
    v33 = v20;
    v35 = v16;
    v23 = sub_1F58D20((__int64)v40);
    v16 = v35;
    v22 = 0;
    v17 = v32;
    v20 = v33;
    if ( v23 )
    {
      v29 = sub_1F596B0((__int64)v40);
      v17 = v32;
      v20 = v33;
      v21 = v30;
      v16 = v35;
      v22 = v29;
    }
  }
  *((_QWORD *)&v31 + 1) = v20;
  *(_QWORD *)&v31 = v17;
  v24 = sub_1D332F0(v16, 106, (__int64)&v38, v22, v21, 0, a3, a4, a5, v7, v9, v31);
  v26 = sub_1D321C0(
          a1[1],
          (__int64)v24,
          v25,
          (__int64)&v38,
          **(unsigned __int8 **)(a2 + 40),
          *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL),
          a3,
          a4,
          *(double *)a5.m128i_i64);
  if ( v38 )
    sub_161E7C0((__int64)&v38, v38);
  return v26;
}
