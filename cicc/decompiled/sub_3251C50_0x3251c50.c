// Function: sub_3251C50
// Address: 0x3251c50
//
__int64 __fastcall sub_3251C50(__int64 a1, __int64 a2)
{
  __int64 v4; // r13
  __int64 v5; // rdi
  char v6; // r14
  __m128i v7; // rax
  unsigned __int16 v8; // r15
  unsigned __int8 v9; // al
  __int64 v10; // rsi
  __int64 v11; // rbx
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdx
  __int128 v17; // xmm0
  __int64 v18; // rax
  __int128 v19; // xmm1
  __int64 v21; // [rsp+38h] [rbp-E8h]
  _QWORD v22[2]; // [rsp+40h] [rbp-E0h] BYREF
  _QWORD v23[2]; // [rsp+50h] [rbp-D0h] BYREF
  unsigned int v24; // [rsp+60h] [rbp-C0h] BYREF
  char v25; // [rsp+68h] [rbp-B8h]
  __m128i v26; // [rsp+70h] [rbp-B0h] BYREF
  char v27; // [rsp+80h] [rbp-A0h]
  __m128i v28; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v29; // [rsp+C0h] [rbp-60h]
  __int128 v30; // [rsp+D0h] [rbp-50h]
  __int64 v31; // [rsp+E0h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 416);
  if ( !v4 )
    return sub_373B2C0(*(_QWORD *)(a1 + 408), a2);
  if ( !*(_BYTE *)(a1 + 424) )
  {
    *(_BYTE *)(a1 + 424) = 1;
    sub_3249C40((__int64 *)a1, a1 + 8, 16, 0);
    v4 = *(_QWORD *)(a1 + 416);
  }
  v5 = *(_QWORD *)(a2 + 40);
  v6 = 0;
  if ( v5 )
  {
    v7.m128i_i64[0] = sub_B91420(v5);
    v6 = 1;
    v28 = v7;
  }
  v8 = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 184) + 216LL) + 1904LL);
  sub_3222AF0(&v26, *(_QWORD *)(a1 + 208), a2);
  v9 = *(_BYTE *)(a2 - 16);
  v10 = a2 - 16;
  if ( (v9 & 2) == 0 )
  {
    v11 = *(_QWORD *)(v10 - 8LL * ((v9 >> 2) & 0xF));
    if ( !v11 )
    {
      v13 = 0;
      goto LABEL_16;
    }
LABEL_8:
    v10 = a2 - 16;
    v11 = sub_B91420(v11);
    v9 = *(_BYTE *)(a2 - 16);
    v13 = v12;
    if ( (v9 & 2) != 0 )
      goto LABEL_9;
LABEL_16:
    v14 = *(_QWORD *)(v10 - 8LL * ((v9 >> 2) & 0xF) + 8);
    if ( v14 )
      goto LABEL_10;
LABEL_17:
    v16 = 0;
    goto LABEL_11;
  }
  v11 = **(_QWORD **)(a2 - 32);
  if ( v11 )
    goto LABEL_8;
  v13 = 0;
LABEL_9:
  v14 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
  if ( !v14 )
    goto LABEL_17;
LABEL_10:
  v21 = v13;
  v15 = sub_B91420(v14);
  v13 = v21;
  v14 = v15;
LABEL_11:
  LOBYTE(v29) = v6;
  v17 = (__int128)_mm_loadu_si128(&v28);
  v18 = v29;
  *(_BYTE *)(v4 + 520) = 1;
  v30 = v17;
  v19 = (__int128)_mm_loadu_si128(&v26);
  v31 = v18;
  v22[0] = v14;
  v22[1] = v16;
  v23[1] = v13;
  v23[0] = v11;
  sub_E78AD0((__int64)&v24, v4, (__int64)v22, (__int64)v23, v8, 0, v19, v27, v17, v18);
  if ( (v25 & 1) != 0 )
    BUG();
  return v24;
}
