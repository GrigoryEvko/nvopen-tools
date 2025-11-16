// Function: sub_213B3C0
// Address: 0x213b3c0
//
__int64 __fastcall sub_213B3C0(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v6; // rsi
  const void **v7; // r13
  unsigned int v8; // r14d
  __int64 v9; // rsi
  unsigned __int8 *v10; // rax
  __int64 v11; // rsi
  __int64 result; // rax
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rcx
  unsigned __int64 v18; // r11
  __int64 v19; // r10
  __int64 v20; // rax
  __int64 *v21; // r13
  __int64 v22; // rax
  unsigned __int8 v23; // bl
  __int64 v24; // r14
  __int64 *v25; // r12
  unsigned __int8 *v26; // rax
  __int128 v27; // rax
  bool v28; // al
  char v29; // al
  __int64 v30; // rdx
  unsigned __int8 v31; // [rsp+0h] [rbp-70h]
  __int64 v32; // [rsp+0h] [rbp-70h]
  __int64 v33; // [rsp+0h] [rbp-70h]
  __int64 v34; // [rsp+0h] [rbp-70h]
  unsigned __int64 v35; // [rsp+8h] [rbp-68h]
  unsigned __int64 v36; // [rsp+8h] [rbp-68h]
  __int64 v37; // [rsp+10h] [rbp-60h] BYREF
  int v38; // [rsp+18h] [rbp-58h]
  unsigned __int8 v39[8]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v40; // [rsp+28h] [rbp-48h]
  const void **v41; // [rsp+30h] [rbp-40h]

  sub_1F40D10(
    (__int64)v39,
    *(_QWORD *)a1,
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = *(_QWORD *)(a2 + 72);
  v7 = v41;
  v31 = v40;
  v8 = (unsigned __int8)v40;
  v37 = v6;
  if ( v6 )
    sub_1623A60((__int64)&v37, v6, 2);
  v9 = *(_QWORD *)a1;
  v38 = *(_DWORD *)(a2 + 64);
  v10 = (unsigned __int8 *)(*(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL)
                          + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL));
  sub_1F40D10((__int64)v39, v9, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), *v10, *((_QWORD *)v10 + 1));
  if ( v39[0] != 1 )
  {
    v11 = *(unsigned __int16 *)(a2 + 24);
LABEL_5:
    result = sub_1D309E0(
               *(__int64 **)(a1 + 8),
               v11,
               (__int64)&v37,
               v8,
               v7,
               0,
               *(double *)a3.m128i_i64,
               a4,
               *(double *)a5.m128i_i64,
               *(_OWORD *)*(_QWORD *)(a2 + 32));
    goto LABEL_6;
  }
  v13 = sub_2138AD0(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v17 = v31;
  v11 = *(unsigned __int16 *)(a2 + 24);
  v18 = v14;
  v19 = v13;
  v20 = *(_QWORD *)(v13 + 40) + 16LL * (unsigned int)v14;
  if ( v31 != *(_BYTE *)v20 || *(const void ***)(v20 + 8) != v7 && !v31 )
    goto LABEL_5;
  if ( (_DWORD)v11 == 142 )
  {
    v25 = *(__int64 **)(a1 + 8);
    v33 = v19;
    v35 = v14;
    v26 = (unsigned __int8 *)(*(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL)
                            + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL));
    *(_QWORD *)&v27 = sub_1D2EF30(v25, *v26, *((_QWORD *)v26 + 1), v17, v15, v16);
    result = (__int64)sub_1D332F0(v25, 148, (__int64)&v37, v8, v7, 0, *(double *)a3.m128i_i64, a4, a5, v33, v35, v27);
  }
  else
  {
    result = v19;
    if ( (_DWORD)v11 == 143 )
    {
      v21 = *(__int64 **)(a1 + 8);
      v22 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL);
      v23 = *(_BYTE *)v22;
      v24 = *(_QWORD *)(v22 + 8);
      v39[0] = v23;
      v40 = v24;
      if ( v23 )
      {
        if ( (unsigned __int8)(v23 - 14) <= 0x5Fu )
        {
          switch ( v23 )
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
              v23 = 3;
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
              v23 = 4;
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
              v23 = 5;
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
              v23 = 6;
              break;
            case 0x37u:
              v23 = 7;
              break;
            case 0x56u:
            case 0x57u:
            case 0x58u:
            case 0x62u:
            case 0x63u:
            case 0x64u:
              v23 = 8;
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
              v23 = 9;
              break;
            case 0x5Eu:
            case 0x5Fu:
            case 0x60u:
            case 0x61u:
            case 0x6Au:
            case 0x6Bu:
            case 0x6Cu:
            case 0x6Du:
              v23 = 10;
              break;
            default:
              v23 = 2;
              break;
          }
          v24 = 0;
        }
      }
      else
      {
        v34 = v19;
        v36 = v14;
        v28 = sub_1F58D20((__int64)v39);
        v19 = v34;
        v18 = v36;
        if ( v28 )
        {
          v29 = sub_1F596B0((__int64)v39);
          v19 = v34;
          v18 = v36;
          v23 = v29;
          v24 = v30;
        }
      }
      result = (__int64)sub_1D3BC50(v21, v19, v18, (__int64)&v37, v23, v24, a3, a4, a5);
    }
  }
LABEL_6:
  if ( v37 )
  {
    v32 = result;
    sub_161E7C0((__int64)&v37, v37);
    return v32;
  }
  return result;
}
