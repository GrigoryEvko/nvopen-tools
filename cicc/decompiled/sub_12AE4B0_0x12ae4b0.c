// Function: sub_12AE4B0
// Address: 0x12ae4b0
//
__int64 __fastcall sub_12AE4B0(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rax
  int *v4; // rax
  int v5; // eax
  _QWORD *v6; // r13
  __int64 v7; // rcx
  __m128i *v8; // rax
  __int64 v9; // rcx
  __m128i *v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 *v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rax
  __m128i *v20; // rdi
  int v22; // [rsp+30h] [rbp-100h]
  _QWORD v23[2]; // [rsp+40h] [rbp-F0h] BYREF
  __int16 v24; // [rsp+50h] [rbp-E0h]
  _QWORD v25[2]; // [rsp+60h] [rbp-D0h] BYREF
  _QWORD v26[2]; // [rsp+70h] [rbp-C0h] BYREF
  __m128i *v27; // [rsp+80h] [rbp-B0h]
  __int64 v28; // [rsp+88h] [rbp-A8h]
  __m128i v29; // [rsp+90h] [rbp-A0h] BYREF
  _QWORD *v30; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v31; // [rsp+A8h] [rbp-88h]
  _QWORD v32[2]; // [rsp+B0h] [rbp-80h] BYREF
  __m128i *v33; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v34; // [rsp+C8h] [rbp-68h]
  __m128i v35; // [rsp+D0h] [rbp-60h] BYREF
  _OWORD *v36; // [rsp+E0h] [rbp-50h] BYREF
  __int64 v37; // [rsp+E8h] [rbp-48h]
  _OWORD v38[4]; // [rsp+F0h] [rbp-40h] BYREF

  v3 = sub_1643270(*(_QWORD *)(a2[2] + 40LL));
  LOBYTE(v26[0]) = 0;
  v22 = sub_1644EA0(v3, 0, 0, 0);
  v25[0] = v26;
  v4 = (int *)a2[3];
  v25[1] = 0;
  v5 = *v4;
  if ( v5 == 4 )
  {
    sub_2241130(v25, 0, 0, "acq_rel", 7);
    goto LABEL_5;
  }
  if ( v5 <= 4 )
  {
    if ( (unsigned int)v5 <= 3 )
      goto LABEL_4;
LABEL_31:
    sub_127B630("unexpected memory order.", 0);
  }
  if ( v5 != 5 )
    goto LABEL_31;
LABEL_4:
  sub_2241130(v25, 0, 0, "sc", 2);
LABEL_5:
  v6 = (_QWORD *)a2[4];
  sub_8FD6D0((__int64)&v30, "fence.", v25);
  if ( v31 == 0x3FFFFFFFFFFFFFFFLL )
    goto LABEL_36;
  v8 = (__m128i *)sub_2241490(&v30, ".", 1, v7);
  v33 = &v35;
  if ( (__m128i *)v8->m128i_i64[0] == &v8[1] )
  {
    v35 = _mm_loadu_si128(v8 + 1);
  }
  else
  {
    v33 = (__m128i *)v8->m128i_i64[0];
    v35.m128i_i64[0] = v8[1].m128i_i64[0];
  }
  v34 = v8->m128i_i64[1];
  v9 = v34;
  v8->m128i_i64[0] = (__int64)v8[1].m128i_i64;
  v8->m128i_i64[1] = 0;
  v8[1].m128i_i8[0] = 0;
  v10 = (__m128i *)sub_2241490(&v33, *v6, v6[1], v9);
  v36 = v38;
  if ( (__m128i *)v10->m128i_i64[0] == &v10[1] )
  {
    v38[0] = _mm_loadu_si128(v10 + 1);
  }
  else
  {
    v36 = (_OWORD *)v10->m128i_i64[0];
    *(_QWORD *)&v38[0] = v10[1].m128i_i64[0];
  }
  v11 = v10->m128i_i64[1];
  v37 = v11;
  v10->m128i_i64[0] = (__int64)v10[1].m128i_i64;
  v10->m128i_i64[1] = 0;
  v10[1].m128i_i8[0] = 0;
  if ( v37 == 0x3FFFFFFFFFFFFFFFLL )
LABEL_36:
    sub_4262D8((__int64)"basic_string::append");
  v12 = sub_2241490(&v36, ";", 1, v11);
  v27 = &v29;
  if ( *(_QWORD *)v12 == v12 + 16 )
  {
    v29 = _mm_loadu_si128((const __m128i *)(v12 + 16));
  }
  else
  {
    v27 = *(__m128i **)v12;
    v29.m128i_i64[0] = *(_QWORD *)(v12 + 16);
  }
  v13 = *(_QWORD *)(v12 + 8);
  *(_BYTE *)(v12 + 16) = 0;
  v28 = v13;
  *(_QWORD *)v12 = v12 + 16;
  *(_QWORD *)(v12 + 8) = 0;
  if ( v36 != v38 )
    j_j___libc_free_0(v36, *(_QWORD *)&v38[0] + 1LL);
  if ( v33 != &v35 )
    j_j___libc_free_0(v33, v35.m128i_i64[0] + 1);
  if ( v30 != v32 )
    j_j___libc_free_0(v30, v32[0] + 1LL);
  v14 = (__int64 *)a2[5];
  strcpy((char *)v32, "~{memory}");
  v30 = v32;
  v36 = v38;
  v31 = 9;
  sub_12A72D0((__int64 *)&v36, v32, (__int64)&v32[1] + 1);
  v33 = &v35;
  sub_12A72D0((__int64 *)&v33, v27, (__int64)v27->m128i_i64 + v28);
  v15 = sub_15EE570(v22, (_DWORD)v33, v34, (_DWORD)v36, v37, 1, 0, 0);
  v16 = *v14;
  v24 = 257;
  v17 = sub_1285290((__int64 *)(v16 + 48), *(_QWORD *)(*(_QWORD *)v15 + 24LL), v15, 0, 0, (__int64)v23, 0);
  v23[0] = *(_QWORD *)(v17 + 56);
  v18 = sub_16498A0(v17);
  v19 = sub_1563AB0(v23, v18, 0xFFFFFFFFLL, 30);
  *(_BYTE *)(a1 + 12) &= ~1u;
  v23[0] = v19;
  *(_QWORD *)(v17 + 56) = v19;
  v20 = v33;
  *(_QWORD *)a1 = v17;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  if ( v20 != &v35 )
    j_j___libc_free_0(v20, v35.m128i_i64[0] + 1);
  if ( v36 != v38 )
    j_j___libc_free_0(v36, *(_QWORD *)&v38[0] + 1LL);
  if ( v30 != v32 )
    j_j___libc_free_0(v30, v32[0] + 1LL);
  if ( v27 != &v29 )
    j_j___libc_free_0(v27, v29.m128i_i64[0] + 1);
  if ( (_QWORD *)v25[0] != v26 )
    j_j___libc_free_0(v25[0], v26[0] + 1LL);
  return a1;
}
