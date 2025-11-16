// Function: sub_19FFCB0
// Address: 0x19ffcb0
//
__int64 __fastcall sub_19FFCB0(
        __int64 ***a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 **v9; // r14
  char v10; // al
  __int64 v11; // rsi
  __int64 ***v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r12
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  __int64 ***v18; // rax
  __int64 **v19; // rsi
  unsigned __int64 v20; // rcx
  __int64 v21; // rcx
  __int64 **v22; // rcx
  _QWORD *v23; // rax
  __int64 *v24; // r14
  double v25; // xmm4_8
  double v26; // xmm5_8
  __int64 v27; // rsi
  __int64 v29; // rsi
  unsigned __int8 *v30; // rsi
  __int64 v31[2]; // [rsp+0h] [rbp-40h] BYREF
  __int16 v32; // [rsp+10h] [rbp-30h]

  v9 = *a1;
  v10 = *((_BYTE *)*a1 + 8);
  if ( v10 == 16 )
    v10 = *(_BYTE *)(*v9[2] + 8);
  if ( v10 == 11 )
  {
    v11 = sub_15A04A0(*a1);
  }
  else
  {
    a2 = (__m128)0xBFF0000000000000LL;
    v11 = sub_15A10B0((__int64)*a1, -1.0);
  }
  v32 = 257;
  if ( (*((_BYTE *)a1 + 23) & 0x40) != 0 )
    v12 = (__int64 ***)*(a1 - 1);
  else
    v12 = &a1[-3 * (*((_DWORD *)a1 + 5) & 0xFFFFFFF)];
  v15 = sub_19FE1F0((__int64 *)v12[3], v11, (__int64)v31, (__int64)a1, (__int64)a1);
  v16 = *((unsigned __int8 *)a1 + 16);
  if ( (unsigned __int8)v16 <= 0x2Fu )
  {
    v13 = 0x80A800000000LL;
    if ( _bittest64(&v13, v16) )
    {
      sub_15F2330(v15, (*((_BYTE *)a1 + 17) & 4) != 0);
      v11 = (*((_BYTE *)a1 + 17) & 2) != 0;
      sub_15F2310(v15, v11);
    }
  }
  v17 = sub_15A06D0(v9, v11, v13, v14);
  if ( (*((_BYTE *)a1 + 23) & 0x40) != 0 )
    v18 = (__int64 ***)*(a1 - 1);
  else
    v18 = &a1[-3 * (*((_DWORD *)a1 + 5) & 0xFFFFFFF)];
  if ( v18[3] )
  {
    v19 = v18[4];
    v20 = (unsigned __int64)v18[5] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v20 = v19;
    if ( v19 )
      v19[2] = (__int64 *)((unsigned __int64)v19[2] & 3 | v20);
  }
  v18[3] = (__int64 **)v17;
  if ( v17 )
  {
    v21 = *(_QWORD *)(v17 + 8);
    v18[4] = (__int64 **)v21;
    if ( v21 )
      *(_QWORD *)(v21 + 16) = (unsigned __int64)(v18 + 4) | *(_QWORD *)(v21 + 16) & 3LL;
    v22 = v18[5];
    v23 = v18 + 3;
    v23[2] = (v17 + 8) | (unsigned __int8)v22 & 3;
    *(_QWORD *)(v17 + 8) = v23;
  }
  v24 = (__int64 *)(v15 + 48);
  sub_164B7C0(v15, (__int64)a1);
  sub_164D160((__int64)a1, v15, a2, a3, a4, a5, v25, v26, a8, a9);
  v27 = (__int64)a1[6];
  v31[0] = v27;
  if ( v27 )
  {
    sub_1623A60((__int64)v31, v27, 2);
    if ( v24 == v31 )
    {
      if ( v31[0] )
        sub_161E7C0((__int64)v31, v31[0]);
      return v15;
    }
    v29 = *(_QWORD *)(v15 + 48);
    if ( !v29 )
    {
LABEL_28:
      v30 = (unsigned __int8 *)v31[0];
      *(_QWORD *)(v15 + 48) = v31[0];
      if ( v30 )
      {
        sub_1623210((__int64)v31, v30, v15 + 48);
        return v15;
      }
      return v15;
    }
LABEL_27:
    sub_161E7C0(v15 + 48, v29);
    goto LABEL_28;
  }
  if ( v24 == v31 )
    return v15;
  v29 = *(_QWORD *)(v15 + 48);
  if ( v29 )
    goto LABEL_27;
  return v15;
}
