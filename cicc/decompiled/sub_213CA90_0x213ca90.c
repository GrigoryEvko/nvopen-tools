// Function: sub_213CA90
// Address: 0x213ca90
//
__int64 *__fastcall sub_213CA90(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  unsigned int *v7; // rdx
  __int64 v8; // rax
  char v9; // cl
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // r12
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // r13
  __int64 v17; // rdi
  __int64 v18; // rax
  unsigned int v19; // edx
  unsigned __int8 v20; // al
  char v21; // di
  __int64 v22; // rcx
  unsigned int v23; // eax
  __int64 v24; // rcx
  __int64 v25; // r10
  __int128 v26; // rax
  unsigned int v27; // edx
  __int64 *v28; // r12
  unsigned __int8 v30; // al
  __int128 v31; // [rsp-10h] [rbp-90h]
  __int64 v32; // [rsp+0h] [rbp-80h]
  __int64 v33; // [rsp+8h] [rbp-78h]
  __int64 *v34; // [rsp+8h] [rbp-78h]
  __int64 *v35; // [rsp+10h] [rbp-70h]
  unsigned __int64 v36; // [rsp+18h] [rbp-68h]
  __int64 *v37; // [rsp+20h] [rbp-60h]
  char v38[8]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v39; // [rsp+38h] [rbp-48h]
  __int64 v40; // [rsp+40h] [rbp-40h] BYREF
  int v41; // [rsp+48h] [rbp-38h]

  v7 = *(unsigned int **)(a2 + 32);
  v8 = *(_QWORD *)(*(_QWORD *)v7 + 40LL) + 16LL * v7[2];
  v9 = *(_BYTE *)v8;
  v10 = *(_QWORD *)(v8 + 8);
  v38[0] = v9;
  v39 = v10;
  v35 = sub_2139210(a1, *(_QWORD *)v7, *((_QWORD *)v7 + 1), a3, a4, a5);
  v36 = v11;
  v12 = sub_2138AD0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
  v13 = *(_QWORD *)(a2 + 72);
  v14 = v12;
  v16 = v15;
  v40 = v13;
  if ( v13 )
    sub_1623A60((__int64)&v40, v13, 2);
  v17 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL);
  v33 = *(_QWORD *)(a1 + 8);
  v41 = *(_DWORD *)(a2 + 64);
  v18 = sub_1E0A0C0(v17);
  v19 = 8 * sub_15A9520(v18, 0);
  if ( v19 == 32 )
  {
    v20 = 5;
  }
  else if ( v19 > 0x20 )
  {
    v20 = 6;
    if ( v19 != 64 )
    {
      v30 = 0;
      v21 = v38[0];
      if ( v19 == 128 )
        v30 = 7;
      v22 = v30;
      if ( !v38[0] )
        goto LABEL_8;
      goto LABEL_16;
    }
  }
  else
  {
    v20 = 3;
    if ( v19 != 8 )
      v20 = 4 * (v19 == 16);
  }
  v21 = v38[0];
  v22 = v20;
  if ( !v38[0] )
  {
LABEL_8:
    v32 = v22;
    v23 = sub_1F58D40((__int64)v38);
    v24 = v32;
    v25 = v33;
    goto LABEL_9;
  }
LABEL_16:
  v23 = sub_2127930(v21);
LABEL_9:
  v34 = (__int64 *)v25;
  *(_QWORD *)&v26 = sub_1D38BB0(v25, v23, (__int64)&v40, v24, 0, 0, a3, a4, a5, 0);
  v37 = sub_1D332F0(
          v34,
          122,
          (__int64)&v40,
          **(unsigned __int8 **)(a2 + 40),
          *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL),
          0,
          *(double *)a3.m128i_i64,
          a4,
          a5,
          v14,
          v16,
          v26);
  *((_QWORD *)&v31 + 1) = v27 | v16 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v31 = v37;
  v28 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          119,
          (__int64)&v40,
          **(unsigned __int8 **)(a2 + 40),
          *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL),
          0,
          *(double *)a3.m128i_i64,
          a4,
          a5,
          (__int64)v35,
          v36,
          v31);
  if ( v40 )
    sub_161E7C0((__int64)&v40, v40);
  return v28;
}
