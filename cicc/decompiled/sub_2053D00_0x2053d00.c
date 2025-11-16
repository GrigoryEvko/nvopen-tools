// Function: sub_2053D00
// Address: 0x2053d00
//
void __fastcall sub_2053D00(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // r14
  __int64 v11; // r15
  __int64 v12; // rdi
  int v13; // edx
  __int64 v14; // rax
  unsigned int v15; // edx
  unsigned __int8 v16; // al
  __int64 v17; // r15
  int v18; // edx
  __int64 v19; // rdi
  unsigned int v20; // edx
  unsigned __int8 v21; // al
  __int64 v22; // rax
  __int64 v23; // r13
  int v24; // edx
  int v25; // edx
  __int64 *v26; // rbx
  int v27; // r14d
  __int128 v28; // [rsp-20h] [rbp-F0h]
  __int64 (__fastcall *v29)(__int64, __int64); // [rsp+8h] [rbp-C8h]
  __int64 (__fastcall *v30)(__int64, __int64); // [rsp+8h] [rbp-C8h]
  __int64 v31; // [rsp+40h] [rbp-90h]
  __int64 *v32; // [rsp+50h] [rbp-80h]
  __int64 v33; // [rsp+60h] [rbp-70h] BYREF
  int v34; // [rsp+68h] [rbp-68h]
  __int128 v35; // [rsp+70h] [rbp-60h] BYREF
  __int128 v36; // [rsp+80h] [rbp-50h]
  __int128 v37; // [rsp+90h] [rbp-40h]

  v7 = *(_DWORD *)(a1 + 536);
  v8 = *(_QWORD *)a1;
  v33 = 0;
  v34 = v7;
  if ( v8 )
  {
    if ( &v33 != (__int64 *)(v8 + 48) )
    {
      v9 = *(_QWORD *)(v8 + 48);
      v33 = v9;
      if ( v9 )
        sub_1623A60((__int64)&v33, v9, 2);
    }
  }
  v10 = *(_QWORD *)(*(_QWORD *)(a1 + 552) + 16LL);
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v32 = sub_2051C20((__int64 *)a1, 0.0, a4, a5);
  v11 = *(_QWORD *)(a1 + 552);
  *(_QWORD *)&v35 = v32;
  v12 = *(_QWORD *)(v11 + 32);
  DWORD2(v35) = v13;
  v29 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v10 + 32LL);
  v14 = sub_1E0A0C0(v12);
  if ( v29 == sub_1F3D950 )
  {
    v15 = 8 * sub_15A9520(v14, 0);
    if ( v15 == 32 )
    {
      v16 = 5;
    }
    else if ( v15 > 0x20 )
    {
      v16 = 6;
      if ( v15 != 64 )
      {
        v16 = 0;
        if ( v15 == 128 )
          v16 = 7;
      }
    }
    else
    {
      v16 = 3;
      if ( v15 != 8 )
        v16 = 4 * (v15 == 16);
    }
  }
  else
  {
    v16 = v29(v10, v14);
  }
  v31 = sub_1D38BB0(
          v11,
          (*(unsigned __int16 *)(a2 + 18) >> 1) & 0x3FFF,
          (__int64)&v33,
          v16,
          0,
          0,
          (__m128i)0LL,
          a4,
          a5,
          0);
  v17 = *(_QWORD *)(a1 + 552);
  *(_QWORD *)&v36 = v31;
  DWORD2(v36) = v18;
  v30 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v10 + 32LL);
  v19 = sub_1E0A0C0(*(_QWORD *)(v17 + 32));
  if ( v30 == sub_1F3D950 )
  {
    v20 = 8 * sub_15A9520(v19, 0);
    if ( v20 == 32 )
    {
      v21 = 5;
    }
    else if ( v20 > 0x20 )
    {
      v21 = 6;
      if ( v20 != 64 )
      {
        v21 = 0;
        if ( v20 == 128 )
          v21 = 7;
      }
    }
    else
    {
      v21 = 3;
      if ( v20 != 8 )
        v21 = 4 * (v20 == 16);
    }
  }
  else
  {
    v21 = v30(v10, v19);
  }
  v22 = sub_1D38BB0(v17, *(unsigned __int8 *)(a2 + 56), (__int64)&v33, v21, 0, 0, (__m128i)0LL, a4, a5, 0);
  v23 = *(_QWORD *)(a1 + 552);
  *(_QWORD *)&v37 = v22;
  *((_QWORD *)&v28 + 1) = 3;
  DWORD2(v37) = v24;
  *(_QWORD *)&v28 = &v35;
  v26 = sub_1D359D0((__int64 *)v23, 218, (__int64)&v33, 1, 0, 0, 0.0, a4, a5, v28);
  v27 = v25;
  if ( v26 )
  {
    nullsub_686();
    *(_QWORD *)(v23 + 176) = v26;
    *(_DWORD *)(v23 + 184) = v27;
    sub_1D23870();
  }
  else
  {
    *(_QWORD *)(v23 + 176) = 0;
    *(_DWORD *)(v23 + 184) = v25;
  }
  if ( v33 )
    sub_161E7C0((__int64)&v33, v33);
}
