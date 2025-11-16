// Function: sub_18A7DD0
// Address: 0x18a7dd0
//
__int64 __fastcall sub_18A7DD0(__int64 a1, __int64 a2)
{
  unsigned __int8 v3; // dl
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // r13
  __int64 *v6; // rax
  __int64 v7; // r12
  __int64 v8; // rsi
  bool v9; // zf
  __int64 v10; // rdx
  int v11; // eax
  unsigned int v12; // r13d
  _QWORD *v13; // rbx
  _QWORD *v14; // r12
  __int64 v15; // rax
  unsigned __int64 v17; // rdx
  _QWORD *v18; // r12
  _QWORD *v19; // rbx
  _QWORD *v20; // r12
  _QWORD *v21; // rdi
  __int64 v22; // r12
  __int64 v23; // r12
  _QWORD *v24; // rbx
  _QWORD *v25; // r12
  _QWORD *v26; // rdi
  _QWORD *v27; // [rsp+0h] [rbp-4E0h]
  __int64 v28; // [rsp+8h] [rbp-4D8h]
  __int64 v29; // [rsp+28h] [rbp-4B8h] BYREF
  __m128i v30[2]; // [rsp+30h] [rbp-4B0h] BYREF
  _BYTE v31[60]; // [rsp+50h] [rbp-490h] BYREF
  __int16 v32; // [rsp+8Ch] [rbp-454h]
  _BYTE v33[32]; // [rsp+90h] [rbp-450h] BYREF
  _BYTE v34[64]; // [rsp+B0h] [rbp-430h] BYREF
  _BYTE v35[32]; // [rsp+F0h] [rbp-3F0h] BYREF
  _BYTE v36[64]; // [rsp+110h] [rbp-3D0h] BYREF
  __m128i v37; // [rsp+150h] [rbp-390h] BYREF
  __int64 v38; // [rsp+160h] [rbp-380h]
  __int64 v39; // [rsp+168h] [rbp-378h]
  __int64 v40; // [rsp+170h] [rbp-370h]
  _BYTE *v41; // [rsp+178h] [rbp-368h]
  __int64 v42; // [rsp+180h] [rbp-360h]
  _BYTE v43[32]; // [rsp+188h] [rbp-358h] BYREF
  _BYTE *v44; // [rsp+1A8h] [rbp-338h]
  __int64 v45; // [rsp+1B0h] [rbp-330h]
  _BYTE v46[192]; // [rsp+1B8h] [rbp-328h] BYREF
  _BYTE *v47; // [rsp+278h] [rbp-268h]
  __int64 v48; // [rsp+280h] [rbp-260h]
  _BYTE v49[72]; // [rsp+288h] [rbp-258h] BYREF
  _QWORD v50[2]; // [rsp+2D0h] [rbp-210h] BYREF
  char v51; // [rsp+2E0h] [rbp-200h]
  _QWORD *v52; // [rsp+328h] [rbp-1B8h]
  unsigned int v53; // [rsp+330h] [rbp-1B0h]
  _BYTE v54[424]; // [rsp+338h] [rbp-1A8h] BYREF

  v3 = *(_BYTE *)(a2 + 16);
  if ( v3 <= 0x17u )
  {
    v4 = 0;
    v5 = 0;
    goto LABEL_4;
  }
  if ( v3 == 78 )
  {
    v17 = a2 | 4;
    v5 = a2 | 4;
  }
  else
  {
    v4 = 0;
    v5 = 0;
    if ( v3 != 29 )
    {
LABEL_4:
      v6 = (__int64 *)(v4 - 72);
      goto LABEL_5;
    }
    v17 = a2 & 0xFFFFFFFFFFFFFFFBLL;
    v5 = a2 & 0xFFFFFFFFFFFFFFFBLL;
  }
  v4 = v17 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v17 & 4) == 0 )
    goto LABEL_4;
  v6 = (__int64 *)(v4 - 24);
LABEL_5:
  v7 = *v6;
  v8 = *(_QWORD *)(a2 + 48);
  v9 = *(_BYTE *)(*v6 + 16) == 0;
  v29 = *(_QWORD *)(a2 + 48);
  if ( !v9 )
    v7 = 0;
  if ( v8 )
    sub_1623A60((__int64)&v29, v8, 2);
  v28 = *(_QWORD *)(a2 + 40);
  sub_3851190(v31);
  if ( HIBYTE(v32) )
  {
    LOBYTE(v32) = 1;
  }
  else
  {
    v8 = 257;
    v32 = 257;
  }
  v9 = *(_QWORD *)(a1 + 1072) == 0;
  v51 = 0;
  if ( v9 )
    sub_4263D6(v31, v8, v10);
  v11 = (*(__int64 (__fastcall **)(__int64, __int64))(a1 + 1080))(a1 + 1056, v7);
  if ( (unsigned int)sub_385A210(v5, (unsigned int)v31, v11, (int)a1 + 1024, (unsigned int)v50, 0, 0) == 0x7FFFFFFF )
  {
    v18 = *(_QWORD **)(a1 + 1264);
    sub_15C9090((__int64)&v37, &v29);
    sub_15CA330((__int64)v50, (__int64)"sample-profile", (__int64)"Not inline", 10, &v37, v28);
    sub_15CAB20((__int64)v50, "incompatible inlining", 0x15u);
    sub_143AA50(v18, (__int64)v50);
    v19 = v52;
    v50[0] = &unk_49ECF68;
    v20 = &v52[11 * v53];
    if ( v52 != v20 )
    {
      do
      {
        v20 -= 11;
        v21 = (_QWORD *)v20[4];
        if ( v21 != v20 + 6 )
          j_j___libc_free_0(v21, v20[6] + 1LL);
        if ( (_QWORD *)*v20 != v20 + 2 )
          j_j___libc_free_0(*v20, v20[2] + 1LL);
      }
      while ( v19 != v20 );
      v20 = v52;
    }
    if ( v20 != (_QWORD *)v54 )
      _libc_free((unsigned __int64)v20);
    v12 = 0;
  }
  else
  {
    v37.m128i_i64[0] = 0;
    v37.m128i_i64[1] = a1 + 1024;
    v41 = v43;
    v42 = 0x400000000LL;
    v44 = v46;
    v38 = 0;
    v39 = 0;
    v40 = 0;
    v45 = 0x800000000LL;
    v47 = v49;
    v48 = 0x800000000LL;
    v12 = sub_1ADC640(v5, &v37, 0, 1, 0);
    if ( (_BYTE)v12 )
    {
      v27 = *(_QWORD **)(a1 + 1264);
      sub_15C9090((__int64)v30, &v29);
      sub_15CA330((__int64)v50, (__int64)"sample-profile", (__int64)"HotInline", 9, v30, v28);
      sub_15CAB20((__int64)v50, "inlined hot callee '", 0x14u);
      sub_15C9340((__int64)v33, "Callee", 6u, v7);
      v22 = sub_17C2270((__int64)v50, (__int64)v33);
      sub_15CAB20(v22, "' into '", 8u);
      sub_15C9340((__int64)v35, "Caller", 6u, *(_QWORD *)(v28 + 56));
      v23 = sub_17C2270(v22, (__int64)v35);
      sub_15CAB20(v23, "'", 1u);
      sub_143AA50(v27, v23);
      sub_2240A30(v36);
      sub_2240A30(v35);
      sub_2240A30(v34);
      sub_2240A30(v33);
      v24 = v52;
      v50[0] = &unk_49ECF68;
      v25 = &v52[11 * v53];
      if ( v52 != v25 )
      {
        do
        {
          v25 -= 11;
          v26 = (_QWORD *)v25[4];
          if ( v26 != v25 + 6 )
            j_j___libc_free_0(v26, v25[6] + 1LL);
          if ( (_QWORD *)*v25 != v25 + 2 )
            j_j___libc_free_0(*v25, v25[2] + 1LL);
        }
        while ( v24 != v25 );
        v25 = v52;
      }
      if ( v25 != (_QWORD *)v54 )
        _libc_free((unsigned __int64)v25);
    }
    if ( v47 != v49 )
      _libc_free((unsigned __int64)v47);
    v13 = v44;
    v14 = &v44[24 * (unsigned int)v45];
    if ( v44 != (_BYTE *)v14 )
    {
      do
      {
        v15 = *(v14 - 1);
        v14 -= 3;
        if ( v15 != 0 && v15 != -8 && v15 != -16 )
          sub_1649B30(v14);
      }
      while ( v13 != v14 );
      v14 = v44;
    }
    if ( v14 != (_QWORD *)v46 )
      _libc_free((unsigned __int64)v14);
    if ( v41 != v43 )
      _libc_free((unsigned __int64)v41);
  }
  if ( v29 )
    sub_161E7C0((__int64)&v29, v29);
  return v12;
}
