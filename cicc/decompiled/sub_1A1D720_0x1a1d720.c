// Function: sub_1A1D720
// Address: 0x1a1d720
//
__int64 __fastcall sub_1A1D720(__int64 *a1, _BYTE *a2, __int64 **a3, unsigned __int64 a4, const __m128i *a5)
{
  __int64 v8; // rax
  __int64 v9; // rax
  _QWORD *v10; // rax
  _QWORD *v11; // r12
  __int64 v12; // rax
  __int64 *v13; // rax
  __int64 *v14; // rax
  int v15; // r8d
  __int64 *v16; // r10
  __int64 **v17; // rcx
  __int64 **v18; // rax
  __int64 v19; // rdx
  __int64 *v20; // rax
  __int64 v21; // r13
  unsigned __int64 *v22; // r15
  __m128i v23; // xmm0
  unsigned __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rsi
  unsigned __int8 *v28; // rsi
  __int64 *v30; // rax
  int v31; // [rsp+Ch] [rbp-B4h]
  __int64 v32; // [rsp+18h] [rbp-A8h]
  __int64 v33; // [rsp+20h] [rbp-A0h]
  char v34[16]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v35; // [rsp+40h] [rbp-80h]
  __m128i v36; // [rsp+50h] [rbp-70h] BYREF
  __int64 v37; // [rsp+60h] [rbp-60h]
  __m128i v38; // [rsp+70h] [rbp-50h] BYREF
  __int16 v39; // [rsp+80h] [rbp-40h]

  if ( a2[16] <= 0x10u )
  {
    if ( !a4 )
    {
LABEL_30:
      v38.m128i_i8[4] = 0;
      return sub_15A2E80(0, (__int64)a2, a3, a4, 1u, (__int64)&v38, 0);
    }
    v8 = 0;
    while ( *((_BYTE *)a3[v8] + 16) <= 0x10u )
    {
      if ( ++v8 == a4 )
        goto LABEL_30;
    }
  }
  v9 = *(_QWORD *)a2;
  v35 = 257;
  if ( *(_BYTE *)(v9 + 8) == 16 )
    v9 = **(_QWORD **)(v9 + 16);
  v33 = *(_QWORD *)(v9 + 24);
  v10 = sub_1648A60(72, (int)a4 + 1);
  v11 = v10;
  if ( v10 )
  {
    v32 = (__int64)v10;
    v12 = *(_QWORD *)a2;
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
      v12 = **(_QWORD **)(v12 + 16);
    v31 = *(_DWORD *)(v12 + 8) >> 8;
    v13 = (__int64 *)sub_15F9F50(v33, (__int64)a3, a4);
    v14 = (__int64 *)sub_1646BA0(v13, v31);
    v15 = a4 + 1;
    v16 = v14;
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
    {
      v30 = sub_16463B0(v14, *(_QWORD *)(*(_QWORD *)a2 + 32LL));
      v15 = a4 + 1;
      v16 = v30;
    }
    else
    {
      v17 = &a3[a4];
      if ( v17 != a3 )
      {
        v18 = a3;
        while ( 1 )
        {
          v19 = **v18;
          if ( *(_BYTE *)(v19 + 8) == 16 )
            break;
          if ( v17 == ++v18 )
            goto LABEL_17;
        }
        v20 = sub_16463B0(v16, *(_QWORD *)(v19 + 32));
        v15 = a4 + 1;
        v16 = v20;
      }
    }
LABEL_17:
    sub_15F1EA0((__int64)v11, (__int64)v16, 32, (__int64)&v11[-3 * (unsigned int)(a4 + 1)], v15, 0);
    v11[7] = v33;
    v11[8] = sub_15F9F50(v33, (__int64)a3, a4);
    sub_15F9CE0((__int64)v11, (__int64)a2, (__int64 *)a3, a4, (__int64)v34);
  }
  else
  {
    v32 = 0;
  }
  sub_15FA2E0((__int64)v11, 1);
  v21 = a1[1];
  v22 = (unsigned __int64 *)a1[2];
  if ( a5[1].m128i_i8[0] > 1u )
  {
    v39 = 260;
    v38.m128i_i64[0] = (__int64)(a1 + 8);
    sub_14EC200(&v36, &v38, a5);
  }
  else
  {
    v23 = _mm_loadu_si128(a5);
    v37 = a5[1].m128i_i64[0];
    v36 = v23;
  }
  if ( v21 )
  {
    sub_157E9D0(v21 + 40, (__int64)v11);
    v24 = *v22;
    v25 = v11[3];
    v11[4] = v22;
    v24 &= 0xFFFFFFFFFFFFFFF8LL;
    v11[3] = v24 | v25 & 7;
    *(_QWORD *)(v24 + 8) = v11 + 3;
    *v22 = *v22 & 7 | (unsigned __int64)(v11 + 3);
  }
  sub_164B780(v32, v36.m128i_i64);
  v26 = *a1;
  if ( *a1 )
  {
    v38.m128i_i64[0] = *a1;
    sub_1623A60((__int64)&v38, v26, 2);
    v27 = v11[6];
    if ( v27 )
      sub_161E7C0((__int64)(v11 + 6), v27);
    v28 = (unsigned __int8 *)v38.m128i_i64[0];
    v11[6] = v38.m128i_i64[0];
    if ( v28 )
      sub_1623210((__int64)&v38, v28, (__int64)(v11 + 6));
  }
  return (__int64)v11;
}
