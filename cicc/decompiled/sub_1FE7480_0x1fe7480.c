// Function: sub_1FE7480
// Address: 0x1fe7480
//
__int64 __fastcall sub_1FE7480(__int64 *a1, int *a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rsi
  _QWORD *v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rdi
  int v14; // eax
  __int64 v15; // rax
  char v16; // dl
  unsigned int v17; // edx
  __int64 v18; // rdi
  __int64 v19; // rsi
  unsigned int v20; // r10d
  __int64 v21; // rax
  unsigned __int64 v22; // r11
  __int64 v23; // r8
  __int64 v24; // rdx
  __int64 v25; // r14
  _QWORD *v27; // rax
  int v28; // edx
  __int32 v29; // eax
  __int64 v30; // rax
  int v31; // [rsp+0h] [rbp-A0h]
  __int64 v32; // [rsp+8h] [rbp-98h]
  unsigned int v33; // [rsp+8h] [rbp-98h]
  __int64 v34; // [rsp+10h] [rbp-90h]
  __int64 v35; // [rsp+18h] [rbp-88h]
  __int64 v36; // [rsp+28h] [rbp-78h] BYREF
  __int64 v37; // [rsp+30h] [rbp-70h] BYREF
  _QWORD *v38; // [rsp+38h] [rbp-68h]
  __m128i v39; // [rsp+40h] [rbp-60h] BYREF
  __int64 v40; // [rsp+50h] [rbp-50h]
  __int64 v41; // [rsp+58h] [rbp-48h]
  __int64 v42; // [rsp+60h] [rbp-40h]

  v35 = *((_QWORD *)a2 + 2);
  v6 = *((_QWORD *)a2 + 3);
  v7 = *((_QWORD *)a2 + 4);
  v34 = v6;
  v36 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v36, v7, 2);
  v8 = a1[2];
  v9 = *a1;
  v10 = *(_QWORD *)(v8 + 8) + 768LL;
  if ( a2[11] == 2 )
  {
    v27 = sub_1E0B640(*a1, v10, &v36, 0);
    v28 = *a2;
    v39.m128i_i64[0] = 5;
    v40 = 0;
    v25 = (__int64)v27;
    LODWORD(v41) = v28;
    sub_1E1A9C0((__int64)v27, v9, &v39);
    if ( *((_BYTE *)a2 + 48) )
    {
      v39.m128i_i64[0] = 1;
      v40 = 0;
      v41 = 0;
    }
    else
    {
      v39 = 0u;
      v40 = 0;
      v41 = 0;
      v42 = 0;
    }
    sub_1E1A9C0(v25, v9, &v39);
    v39.m128i_i64[0] = 14;
    v41 = v35;
    v40 = 0;
    sub_1E1A9C0(v25, v9, &v39);
    v39.m128i_i64[0] = 14;
    v40 = 0;
    v41 = v34;
    sub_1E1A9C0(v25, v9, &v39);
    goto LABEL_22;
  }
  v32 = *(_QWORD *)(v8 + 8) + 768LL;
  v11 = sub_1E0B640(*a1, v10, &v36, 0);
  v37 = v9;
  v12 = v32;
  v38 = v11;
  v13 = (__int64)v11;
  v14 = a2[11];
  if ( !v14 )
  {
    v20 = a2[2];
    v21 = *(unsigned int *)(a3 + 24);
    v22 = *(_QWORD *)a2;
    if ( !(_DWORD)v21 )
      goto LABEL_18;
    v23 = *(_QWORD *)(a3 + 8);
    v31 = 1;
    v33 = (v21 - 1) & (v20 + ((v22 >> 9) ^ (v22 >> 4)));
    while ( 1 )
    {
      v24 = v23 + 24LL * v33;
      if ( v22 == *(_QWORD *)v24 && a2[2] == *(_DWORD *)(v24 + 8) )
        break;
      if ( !*(_QWORD *)v24 && *(_DWORD *)(v24 + 8) == -1 )
        goto LABEL_18;
      v33 = (v21 - 1) & (v31 + v33);
      ++v31;
    }
    if ( v24 == v23 + 24 * v21 )
      goto LABEL_18;
    sub_1FE6BA0(a1, &v37, v22, v20, *(_DWORD *)(v13 + 40), v12, a3, 1, 0, 0);
LABEL_19:
    v18 = (__int64)v38;
    v19 = v37;
    if ( !*((_BYTE *)a2 + 48) )
      goto LABEL_11;
LABEL_20:
    v39.m128i_i64[0] = 1;
    v40 = 0;
    v41 = 0;
    sub_1E1A9C0(v18, v19, &v39);
    goto LABEL_21;
  }
  if ( v14 == 3 )
  {
    v29 = *a2;
    v40 = 0;
    v39.m128i_i32[2] = v29;
    v41 = 0;
    v42 = 0;
    v39.m128i_i64[0] = 0x800000000LL;
    sub_1E1A9C0(v13, v9, &v39);
    goto LABEL_19;
  }
  if ( v14 != 1 )
  {
LABEL_18:
    v39 = 0u;
    v40 = 0;
    v41 = 0;
    v42 = 0;
    sub_1E1A9C0(v13, v9, &v39);
    goto LABEL_19;
  }
  v15 = *(_QWORD *)a2;
  v16 = *(_BYTE *)(*(_QWORD *)a2 + 16LL);
  if ( v16 != 13 )
  {
    if ( v16 == 14 )
    {
      v39.m128i_i64[0] = 3;
      goto LABEL_10;
    }
    goto LABEL_18;
  }
  v17 = *(_DWORD *)(v15 + 32);
  if ( v17 <= 0x40 )
  {
    v30 = *(_QWORD *)(v15 + 24);
    v39.m128i_i64[0] = 1;
    v15 = v30 << (64 - (unsigned __int8)v17) >> (64 - (unsigned __int8)v17);
  }
  else
  {
    v39.m128i_i64[0] = 2;
  }
LABEL_10:
  v40 = 0;
  v41 = v15;
  sub_1E1A9C0(v13, v9, &v39);
  v18 = (__int64)v38;
  v19 = v37;
  if ( *((_BYTE *)a2 + 48) )
    goto LABEL_20;
LABEL_11:
  v39 = (__m128i)0x800000000uLL;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  sub_1E1A9C0(v18, v19, &v39);
LABEL_21:
  v39.m128i_i64[0] = 14;
  v40 = 0;
  v41 = v35;
  sub_1E1A9C0((__int64)v38, v37, &v39);
  v39.m128i_i64[0] = 14;
  v40 = 0;
  v41 = v34;
  sub_1E1A9C0((__int64)v38, v37, &v39);
  v25 = (__int64)v38;
LABEL_22:
  if ( v36 )
    sub_161E7C0((__int64)&v36, v36);
  return v25;
}
