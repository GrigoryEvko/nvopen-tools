// Function: sub_1A06A90
// Address: 0x1a06a90
//
__int64 __fastcall sub_1A06A90(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10,
        __int64 a11,
        __int64 a12)
{
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 **v14; // rax
  __int64 v15; // r12
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 *v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rdx
  _QWORD *v23; // rax
  __int64 v24; // rsi
  unsigned __int64 v25; // rcx
  __int64 v26; // rcx
  __int64 v27; // rcx
  _QWORD *v28; // rax
  __int64 *v29; // r14
  double v30; // xmm4_8
  double v31; // xmm5_8
  __int64 v32; // rsi
  __int64 v34; // rsi
  unsigned __int8 *v35; // rsi
  __int64 v36[2]; // [rsp+0h] [rbp-40h] BYREF
  __int16 v37; // [rsp+10h] [rbp-30h]

  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
  {
    v12 = *(_QWORD *)(a1 - 8);
  }
  else
  {
    a12 = 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
    v12 = a1 - a12;
  }
  v13 = sub_1A066E0(*(_QWORD *)(v12 + 24), a1, a2, a12, *(double *)a3.m128_u64, a4, a5);
  v37 = 257;
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v14 = *(__int64 ***)(a1 - 8);
  else
    v14 = (__int64 **)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  v15 = sub_19FE280(*v14, v13, (__int64)v36, a1, a1);
  v18 = sub_15A06D0(*(__int64 ***)a1, v13, v16, v17);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
  {
    v20 = *(__int64 **)(a1 - 8);
  }
  else
  {
    v19 = 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
    v20 = (__int64 *)(a1 - v19);
  }
  if ( *v20 )
  {
    v13 = v20[1];
    v19 = v20[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v19 = v13;
    if ( v13 )
    {
      v19 |= *(_QWORD *)(v13 + 16) & 3LL;
      *(_QWORD *)(v13 + 16) = v19;
    }
  }
  *v20 = v18;
  if ( v18 )
  {
    v21 = *(_QWORD *)(v18 + 8);
    v20[1] = v21;
    if ( v21 )
    {
      v13 = (unsigned __int64)(v20 + 1) | *(_QWORD *)(v21 + 16) & 3LL;
      *(_QWORD *)(v21 + 16) = v13;
    }
    v19 = (v18 + 8) | v20[2] & 3;
    v20[2] = v19;
    *(_QWORD *)(v18 + 8) = v20;
  }
  v22 = sub_15A06D0(*(__int64 ***)a1, v13, (__int64)v20, v19);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v23 = *(_QWORD **)(a1 - 8);
  else
    v23 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  if ( v23[3] )
  {
    v24 = v23[4];
    v25 = v23[5] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v25 = v24;
    if ( v24 )
      *(_QWORD *)(v24 + 16) = *(_QWORD *)(v24 + 16) & 3LL | v25;
  }
  v23[3] = v22;
  if ( v22 )
  {
    v26 = *(_QWORD *)(v22 + 8);
    v23[4] = v26;
    if ( v26 )
      *(_QWORD *)(v26 + 16) = (unsigned __int64)(v23 + 4) | *(_QWORD *)(v26 + 16) & 3LL;
    v27 = v23[5];
    v28 = v23 + 3;
    v28[2] = (v22 + 8) | v27 & 3;
    *(_QWORD *)(v22 + 8) = v28;
  }
  v29 = (__int64 *)(v15 + 48);
  sub_164B7C0(v15, a1);
  sub_164D160(a1, v15, a3, a4, a5, a6, v30, v31, a9, a10);
  v32 = *(_QWORD *)(a1 + 48);
  v36[0] = v32;
  if ( v32 )
  {
    sub_1623A60((__int64)v36, v32, 2);
    if ( v29 == v36 )
    {
      if ( v36[0] )
        sub_161E7C0((__int64)v36, v36[0]);
      return v15;
    }
    v34 = *(_QWORD *)(v15 + 48);
    if ( !v34 )
    {
LABEL_34:
      v35 = (unsigned __int8 *)v36[0];
      *(_QWORD *)(v15 + 48) = v36[0];
      if ( v35 )
      {
        sub_1623210((__int64)v36, v35, v15 + 48);
        return v15;
      }
      return v15;
    }
LABEL_33:
    sub_161E7C0(v15 + 48, v34);
    goto LABEL_34;
  }
  if ( v29 == v36 )
    return v15;
  v34 = *(_QWORD *)(v15 + 48);
  if ( v34 )
    goto LABEL_33;
  return v15;
}
