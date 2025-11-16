// Function: sub_1D40600
// Address: 0x1d40600
//
__int64 __fastcall sub_1D40600(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        const void ***a5,
        const void ***a6,
        __m128i a7,
        double a8,
        __m128i a9)
{
  __int64 v13; // rax
  unsigned int v14; // edx
  unsigned __int8 v15; // al
  __int128 v16; // rax
  int v17; // edx
  __int64 v18; // rax
  unsigned int v19; // edx
  unsigned __int8 v20; // al
  __int64 v21; // rcx
  char v22; // al
  unsigned int v23; // eax
  __int64 v24; // rsi
  __int128 v25; // rax
  int v27; // edx
  unsigned __int8 v28; // al
  __int64 v29; // [rsp+8h] [rbp-58h]
  __int64 (__fastcall *v30)(__int64, __int64); // [rsp+10h] [rbp-50h]
  __int64 v31; // [rsp+10h] [rbp-50h]
  __int64 v32; // [rsp+18h] [rbp-48h]
  int v33; // [rsp+18h] [rbp-48h]
  __int64 (__fastcall *v34)(__int64, __int64); // [rsp+20h] [rbp-40h]
  __int64 *v35; // [rsp+20h] [rbp-40h]

  v32 = a2[2];
  v34 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v32 + 48LL);
  v13 = sub_1E0A0C0(a2[4]);
  if ( v34 == sub_1D13A20 )
  {
    v14 = 8 * sub_15A9520(v13, 0);
    if ( v14 == 32 )
    {
      v15 = 5;
    }
    else if ( v14 > 0x20 )
    {
      v15 = 6;
      if ( v14 != 64 )
      {
        v15 = 0;
        if ( v14 == 128 )
          v15 = 7;
      }
    }
    else
    {
      v15 = 3;
      if ( v14 != 8 )
        v15 = 4 * (v14 == 16);
    }
  }
  else
  {
    v15 = v34(v32, v13);
  }
  *(_QWORD *)&v16 = sub_1D38BB0((__int64)a2, 0, a4, v15, 0, 0, a7, a8, a9, 0);
  v35 = sub_1D332F0(
          a2,
          109,
          a4,
          *(unsigned int *)a5,
          a5[1],
          0,
          *(double *)a7.m128i_i64,
          a8,
          a9,
          *(_QWORD *)a3,
          *(_QWORD *)(a3 + 8),
          v16);
  v33 = v17;
  v29 = a2[2];
  v30 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v29 + 48LL);
  v18 = sub_1E0A0C0(a2[4]);
  if ( v30 == sub_1D13A20 )
  {
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
        v28 = 0;
        if ( v19 == 128 )
          v28 = 7;
        v21 = v28;
        v22 = *(_BYTE *)a5;
        if ( !*(_BYTE *)a5 )
          goto LABEL_12;
LABEL_22:
        v24 = word_42E7700[(unsigned __int8)(v22 - 14)];
        goto LABEL_13;
      }
    }
    else
    {
      v20 = 3;
      if ( v19 != 8 )
        v20 = 4 * (v19 == 16);
    }
  }
  else
  {
    v20 = v30(v29, v18);
  }
  v21 = v20;
  v22 = *(_BYTE *)a5;
  if ( *(_BYTE *)a5 )
    goto LABEL_22;
LABEL_12:
  v31 = v21;
  v23 = sub_1F58D30(a5);
  v21 = v31;
  v24 = v23;
LABEL_13:
  *(_QWORD *)&v25 = sub_1D38BB0((__int64)a2, v24, a4, v21, 0, 0, a7, a8, a9, 0);
  *(_QWORD *)(a1 + 16) = sub_1D332F0(
                           a2,
                           109,
                           a4,
                           *(unsigned int *)a6,
                           a6[1],
                           0,
                           *(double *)a7.m128i_i64,
                           a8,
                           a9,
                           *(_QWORD *)a3,
                           *(_QWORD *)(a3 + 8),
                           v25);
  *(_QWORD *)a1 = v35;
  *(_DWORD *)(a1 + 24) = v27;
  *(_DWORD *)(a1 + 8) = v33;
  return a1;
}
