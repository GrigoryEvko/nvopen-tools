// Function: sub_38B7F60
// Address: 0x38b7f60
//
__int64 __fastcall sub_38B7F60(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r14
  unsigned __int64 v11; // rsi
  int v13; // eax
  int v14; // ecx
  double v15; // xmm4_8
  double v16; // xmm5_8
  char v17; // al
  double v18; // xmm4_8
  double v19; // xmm5_8
  int v20; // eax
  __int64 v21; // r13
  int v22; // eax
  unsigned __int64 v23; // rsi
  unsigned __int8 v24; // [rsp+Fh] [rbp-E1h]
  const char *v25; // [rsp+10h] [rbp-E0h] BYREF
  char v26; // [rsp+20h] [rbp-D0h]
  char v27; // [rsp+21h] [rbp-CFh]
  __int64 v28[2]; // [rsp+30h] [rbp-C0h] BYREF
  char v29; // [rsp+40h] [rbp-B0h]
  char v30; // [rsp+41h] [rbp-AFh]

  v10 = a1 + 8;
  if ( *(_DWORD *)(a1 + 64) == 8 )
  {
    v13 = sub_3887100(a1 + 8);
    v14 = -1;
    *(_DWORD *)(a1 + 64) = v13;
    if ( (*(_BYTE *)(a2 + 23) & 0x20) == 0 )
      v14 = ((__int64)(*(_QWORD *)(a1 + 1008) - *(_QWORD *)(a1 + 1000)) >> 3) - 1;
    sub_3894AE0((__int64)v28, a1, a2, v14);
    v17 = sub_389B540(v28, *(double *)a3.m128_u64, *(double *)a4.m128i_i64, a5, a6, v15, v16, a9, a10);
    if ( !v17 )
    {
      v20 = *(_DWORD *)(a1 + 64);
      v21 = *(_QWORD *)(a1 + 1120);
      *(_QWORD *)(a1 + 1120) = v28;
      if ( v20 == 301 || v20 == 9 )
      {
        v23 = *(_QWORD *)(a1 + 56);
        v27 = 1;
        v25 = "function body requires at least one basic block";
        v26 = 3;
        v17 = sub_38814C0(v10, v23, (__int64)&v25);
      }
      else
      {
        while ( 1 )
        {
          if ( (unsigned __int8)sub_38B20D0(a1, v28, a3, a4, a5, a6, v18, v19, a9, a10) )
          {
LABEL_19:
            v17 = 1;
            goto LABEL_14;
          }
          v22 = *(_DWORD *)(a1 + 64);
          if ( v22 == 301 )
            break;
          if ( v22 == 9 )
            goto LABEL_20;
        }
        while ( v22 != 9 )
        {
          if ( (unsigned __int8)sub_38B7E70(a1, v28, *(double *)a3.m128_u64, *(double *)a4.m128i_i64, a5) )
            goto LABEL_19;
          v22 = *(_DWORD *)(a1 + 64);
        }
LABEL_20:
        *(_DWORD *)(a1 + 64) = sub_3887100(v10);
        v17 = sub_388F860(v28);
      }
LABEL_14:
      *(_QWORD *)(a1 + 1120) = v21;
    }
    v24 = v17;
    sub_388D240(v28, a3, *(double *)a4.m128i_i64, a5, a6, v18, v19, a9, a10);
    return v24;
  }
  else
  {
    v11 = *(_QWORD *)(a1 + 56);
    v30 = 1;
    v28[0] = (__int64)"expected '{' in function body";
    v29 = 3;
    return sub_38814C0(a1 + 8, v11, (__int64)v28);
  }
}
