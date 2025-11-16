// Function: sub_2120760
// Address: 0x2120760
//
__int64 *__fastcall sub_2120760(__int64 *a1, __int64 *a2, unsigned int a3, __m128i a4, double a5, __m128i a6)
{
  __int64 *v6; // rbx
  unsigned __int8 *v7; // rax
  __int64 v8; // r13
  __int64 v9; // rsi
  unsigned int v10; // eax
  unsigned int v11; // r13d
  __int64 v12; // r12
  __int64 v13; // rsi
  __int64 v14; // r12
  __int64 v15; // r12
  __int64 v16; // rdx
  __int64 v17; // r13
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rdx
  __int64 v20; // rsi
  __int64 *v21; // r15
  __int64 v22; // r8
  unsigned __int64 v23; // r9
  __int128 v25; // [rsp-10h] [rbp-90h]
  __int64 v26; // [rsp+0h] [rbp-80h]
  unsigned __int64 v27; // [rsp+0h] [rbp-80h]
  unsigned __int64 v28; // [rsp+8h] [rbp-78h]
  unsigned int v29; // [rsp+10h] [rbp-70h] BYREF
  const void **v30; // [rsp+18h] [rbp-68h]
  unsigned __int64 v31; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v32; // [rsp+28h] [rbp-58h]
  __int64 v33; // [rsp+30h] [rbp-50h] BYREF
  int v34; // [rsp+38h] [rbp-48h]
  const void **v35; // [rsp+40h] [rbp-40h]

  v6 = a2;
  v7 = (unsigned __int8 *)(a2[5] + 16LL * a3);
  v8 = *v7;
  v26 = *((_QWORD *)v7 + 1);
  sub_1F40D10((__int64)&v33, *a1, *(_QWORD *)(a1[1] + 48), (unsigned __int8)v8, v26);
  v9 = *a1;
  if ( (_BYTE)v8 != (_BYTE)v34 )
    goto LABEL_2;
  if ( (const void **)v26 == v35 )
  {
    if ( !(_BYTE)v8 )
      goto LABEL_2;
  }
  else if ( !(_BYTE)v8 )
  {
    goto LABEL_2;
  }
  if ( *(_QWORD *)(v9 + 8 * v8 + 120) )
    return v6;
LABEL_2:
  sub_1F40D10((__int64)&v33, v9, *(_QWORD *)(a1[1] + 48), *(unsigned __int8 *)v6[5], *(_QWORD *)(v6[5] + 8));
  LOBYTE(v29) = v34;
  v30 = v35;
  if ( (_BYTE)v34 )
    v10 = sub_211A7A0(v34);
  else
    v10 = sub_1F58D40((__int64)&v29);
  v11 = v10 - 1;
  v32 = v10;
  v12 = ~(1LL << ((unsigned __int8)v10 - 1));
  if ( v10 > 0x40 )
  {
    sub_16A4EF0((__int64)&v31, -1, 1);
    if ( v32 > 0x40 )
    {
      *(_QWORD *)(v31 + 8LL * (v11 >> 6)) &= v12;
      goto LABEL_7;
    }
  }
  else
  {
    v31 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v10;
  }
  v31 &= v12;
LABEL_7:
  v13 = v6[9];
  v14 = a1[1];
  v33 = v13;
  if ( v13 )
    sub_1623A60((__int64)&v33, v13, 2);
  v34 = *((_DWORD *)v6 + 16);
  v15 = sub_1D38970(v14, (__int64)&v31, (__int64)&v33, v29, v30, 0, a4, a5, a6, 0);
  v17 = v16;
  if ( v33 )
    sub_161E7C0((__int64)&v33, v33);
  v18 = sub_2120330((__int64)a1, *(_QWORD *)v6[4], *(_QWORD *)(v6[4] + 8));
  v20 = v6[9];
  v21 = (__int64 *)a1[1];
  v22 = v18;
  v23 = v19;
  v33 = v20;
  if ( v20 )
  {
    v28 = v19;
    v27 = v18;
    sub_1623A60((__int64)&v33, v20, 2);
    v22 = v27;
    v23 = v28;
  }
  *((_QWORD *)&v25 + 1) = v17;
  *(_QWORD *)&v25 = v15;
  v34 = *((_DWORD *)v6 + 16);
  v6 = sub_1D332F0(v21, 118, (__int64)&v33, v29, v30, 0, *(double *)a4.m128i_i64, a5, a6, v22, v23, v25);
  if ( v33 )
    sub_161E7C0((__int64)&v33, v33);
  if ( v32 > 0x40 && v31 )
    j_j___libc_free_0_0(v31);
  return v6;
}
