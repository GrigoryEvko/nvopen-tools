// Function: sub_23A37B0
// Address: 0x23a37b0
//
unsigned __int64 *__fastcall sub_23A37B0(unsigned __int64 *a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  _QWORD *v9; // rax
  __int64 v11; // rax
  _QWORD *v12; // r8
  _QWORD *v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // r14
  _QWORD *v16; // rax
  __int64 v17; // [rsp+0h] [rbp-130h]
  _QWORD *v18; // [rsp+8h] [rbp-128h]
  __int64 v19; // [rsp+18h] [rbp-118h] BYREF
  __int64 v20[4]; // [rsp+20h] [rbp-110h] BYREF
  __int64 v21; // [rsp+40h] [rbp-F0h] BYREF
  _BYTE v22[56]; // [rsp+48h] [rbp-E8h] BYREF
  __m128i v23; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v24; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v25; // [rsp+98h] [rbp-98h]
  __int64 v26; // [rsp+A0h] [rbp-90h]
  int v27; // [rsp+C0h] [rbp-70h]
  __int64 v28; // [rsp+C8h] [rbp-68h]
  __int64 v29; // [rsp+D0h] [rbp-60h]
  __int64 v30; // [rsp+D8h] [rbp-58h]
  __int64 v31; // [rsp+E0h] [rbp-50h]
  __int64 v32; // [rsp+E8h] [rbp-48h]
  __int64 v33; // [rsp+F0h] [rbp-40h]

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  a1[3] = 0;
  a1[4] = 0;
  if ( *(_BYTE *)(a2 + 192) )
  {
    if ( !*(_BYTE *)(a2 + 181) )
      goto LABEL_3;
    v15 = *(_QWORD *)a2;
    v16 = (_QWORD *)sub_22077B0(0x10u);
    if ( v16 )
    {
      v16[1] = v15;
      *v16 = &unk_4A0DF38;
    }
    v23.m128i_i64[0] = (__int64)v16;
    sub_23A2230(a1, (unsigned __int64 *)&v23);
    sub_23501E0(v23.m128i_i64);
    if ( *(_BYTE *)(a2 + 192) )
    {
LABEL_3:
      if ( (unsigned int)(*(_DWORD *)(a2 + 168) - 1) <= 1 )
      {
        v6 = *(_QWORD *)(a2 + 184);
        v20[0] = v6;
        if ( v6 )
          _InterlockedAdd((volatile signed __int32 *)(v6 + 8), 1u);
        sub_2241BD0(v23.m128i_i64, a2 + 104);
        sub_2241BD0(&v21, a2 + 40);
        sub_23A27F0(
          a2,
          a1,
          *(_DWORD *)(a2 + 168) == 1,
          0,
          *(_BYTE *)(a2 + 182),
          (unsigned __int64 *)&v21,
          (__int64)&v23,
          v20);
        sub_2240A30((unsigned __int64 *)&v21);
        sub_2240A30((unsigned __int64 *)&v23);
        if ( v20[0] )
          sub_23569D0((volatile signed __int32 *)(v20[0] + 8));
      }
    }
  }
  v7 = sub_22077B0(0x10u);
  if ( v7 )
  {
    *(_BYTE *)(v7 + 8) = 0;
    *(_QWORD *)v7 = &unk_4A118F8;
  }
  v23.m128i_i64[0] = v7;
  v23.m128i_i8[8] = 0;
  sub_23571D0(a1, v23.m128i_i64);
  sub_233EFE0(v23.m128i_i64);
  sub_23A1280(a2, (__int64)a1, a3);
  if ( *(_BYTE *)(a2 + 192) )
  {
    if ( !*(_BYTE *)(a2 + 180) )
    {
      if ( *(_DWORD *)(a2 + 168) != 3 )
        goto LABEL_13;
      goto LABEL_66;
    }
    v14 = (_QWORD *)sub_22077B0(0x10u);
    if ( v14 )
      *v14 = &unk_4A0ED78;
    v23.m128i_i64[0] = (__int64)v14;
    v23.m128i_i8[8] = 0;
    sub_23571D0(a1, v23.m128i_i64);
    sub_233EFE0(v23.m128i_i64);
    if ( *(_BYTE *)(a2 + 192) && *(_DWORD *)(a2 + 168) == 3 )
    {
LABEL_66:
      v19 = 0;
      sub_2241BD0(&v21, a2 + 104);
      sub_2241BD0(v20, a2 + 40);
      sub_26C1D00((unsigned int)&v23, (unsigned int)v20, (unsigned int)&v21, 0, (unsigned int)&v19, 1, 1);
      sub_2357AD0(a1, &v23);
      sub_233AA80((unsigned __int64 *)&v23);
      sub_2240A30((unsigned __int64 *)v20);
      sub_2240A30((unsigned __int64 *)&v21);
      if ( v19 )
        sub_23569D0((volatile signed __int32 *)(v19 + 8));
      sub_23A2700(a1);
    }
  }
LABEL_13:
  sub_23A12F0(a2, (__int64)a1, a3, a4);
  v8 = sub_22077B0(0x10u);
  if ( v8 )
  {
    *(_BYTE *)(v8 + 8) = 0;
    *(_QWORD *)v8 = &unk_4A0CDB8;
  }
  v23.m128i_i64[0] = v8;
  sub_23A2230(a1, (unsigned __int64 *)&v23);
  sub_23501E0(v23.m128i_i64);
  if ( *(_BYTE *)(a2 + 26) )
  {
    v13 = (_QWORD *)sub_22077B0(0x10u);
    if ( v13 )
      *v13 = &unk_4A0D8F8;
    v23.m128i_i64[0] = (__int64)v13;
    sub_23A2230(a1, (unsigned __int64 *)&v23);
    sub_23501E0(v23.m128i_i64);
  }
  if ( byte_4FDCA48 )
  {
    v11 = sub_22077B0(0x10u);
    if ( v11 )
    {
      *(_BYTE *)(v11 + 8) = 1;
      *(_QWORD *)v11 = &unk_4A11A78;
    }
    v23.m128i_i64[0] = v11;
    v23.m128i_i8[8] = 0;
    sub_23571D0(a1, v23.m128i_i64);
    sub_233EFE0(v23.m128i_i64);
    if ( !*(_DWORD *)(a2 + 536) )
    {
LABEL_18:
      if ( !*(_DWORD *)(a2 + 296) )
        goto LABEL_19;
      goto LABEL_39;
    }
  }
  else if ( !*(_DWORD *)(a2 + 536) )
  {
    goto LABEL_18;
  }
  v23 = 0u;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  sub_23A0F30(a2, (__int64)&v23, a3);
  v12 = (_QWORD *)v23.m128i_i64[1];
  if ( v23.m128i_i64[0] != v23.m128i_i64[1] )
  {
    sub_234A9E0(&v21, (unsigned __int64 *)&v23);
    sub_2357280(a1, &v21);
    sub_233F000(&v21);
    v17 = v23.m128i_i64[1];
    v12 = (_QWORD *)v23.m128i_i64[0];
    if ( v23.m128i_i64[1] != v23.m128i_i64[0] )
    {
      do
      {
        if ( *v12 )
        {
          v18 = v12;
          (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v12 + 8LL))(*v12);
          v12 = v18;
        }
        ++v12;
      }
      while ( (_QWORD *)v17 != v12 );
      v12 = (_QWORD *)v23.m128i_i64[0];
    }
  }
  if ( !v12 )
    goto LABEL_18;
  j_j___libc_free_0((unsigned __int64)v12);
  if ( !*(_DWORD *)(a2 + 296) )
  {
LABEL_19:
    if ( !*(_DWORD *)(a2 + 376) )
      goto LABEL_20;
    goto LABEL_42;
  }
LABEL_39:
  v23.m128i_i64[0] = (__int64)&v24;
  v23.m128i_i64[1] = 0x600000000LL;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  sub_23A0DE0(a2, (__int64)&v23, a3);
  if ( v28 != v29 || v32 != v31 )
  {
    sub_23A20C0((__int64)&v21, (__int64)&v23, 0, 0, 0, (__int64)&v21);
    sub_234CFF0((__int64)v20, &v21, 0);
    sub_23571D0(a1, v20);
    sub_233EFE0(v20);
    sub_233F7F0((__int64)v22);
    sub_233F7D0(&v21);
  }
  sub_2337B30((unsigned __int64 *)&v23);
  if ( !*(_DWORD *)(a2 + 376) )
  {
LABEL_20:
    if ( !*(_DWORD *)(a2 + 456) )
      goto LABEL_21;
    goto LABEL_45;
  }
LABEL_42:
  v23.m128i_i64[0] = (__int64)&v24;
  v23.m128i_i64[1] = 0x600000000LL;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  sub_23A0E50(a2, (__int64)&v23, a3);
  if ( v29 != v28 || v32 != v31 )
  {
    sub_23A20C0((__int64)&v21, (__int64)&v23, 0, 0, 0, (__int64)&v21);
    sub_234CFF0((__int64)v20, &v21, 0);
    sub_23571D0(a1, v20);
    sub_233EFE0(v20);
    sub_233F7F0((__int64)v22);
    sub_233F7D0(&v21);
  }
  sub_2337B30((unsigned __int64 *)&v23);
  if ( *(_DWORD *)(a2 + 456) )
  {
LABEL_45:
    v23 = 0u;
    v24 = 0;
    v25 = 0;
    v26 = 0;
    sub_23A0EC0(a2, (__int64)&v23, a3);
    if ( v23.m128i_i64[1] != v23.m128i_i64[0] )
    {
      sub_234AAB0((__int64)&v21, v23.m128i_i64, 0);
      sub_23571D0(a1, &v21);
      sub_233EFE0(&v21);
    }
    sub_233F7F0((__int64)&v23);
  }
LABEL_21:
  sub_23A1080(a2, (__int64)a1, a3, a4);
  if ( *(_DWORD *)(a2 + 616) )
  {
    v23 = 0u;
    v24 = 0;
    v25 = 0;
    v26 = 0;
    sub_23A0FA0(a2, (__int64)&v23, a3);
    if ( v23.m128i_i64[1] != v23.m128i_i64[0] )
    {
      sub_234AAB0((__int64)&v21, v23.m128i_i64, 0);
      sub_23571D0(a1, &v21);
      sub_233EFE0(&v21);
    }
    sub_233F7F0((__int64)&v23);
  }
  if ( *(_DWORD *)(a2 + 696) )
  {
    v23 = 0u;
    v24 = 0;
    v25 = 0;
    v26 = 0;
    sub_23A1010(a2, (__int64)&v23, a3);
    if ( v23.m128i_i64[1] != v23.m128i_i64[0] )
    {
      sub_234AAB0((__int64)&v21, v23.m128i_i64, 0);
      sub_23571D0(a1, &v21);
      sub_233EFE0(&v21);
    }
    sub_233F7F0((__int64)&v23);
  }
  sub_23A3310((__int64)&v23);
  sub_23A2270(a1, (unsigned __int64 *)&v23);
  sub_234A900((__int64)&v23);
  sub_23A1110(a2, (__int64)a1, a3, a4);
  if ( (a4 & 0xFFFFFFFD) == 1 )
    sub_23A2750(a2, a1);
  v9 = (_QWORD *)sub_22077B0(0x10u);
  if ( v9 )
    *v9 = &unk_4A0EE38;
  v23.m128i_i64[0] = (__int64)v9;
  v23.m128i_i8[8] = 0;
  sub_23571D0(a1, v23.m128i_i64);
  sub_233EFE0(v23.m128i_i64);
  return a1;
}
