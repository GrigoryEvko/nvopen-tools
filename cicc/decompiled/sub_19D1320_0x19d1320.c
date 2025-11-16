// Function: sub_19D1320
// Address: 0x19d1320
//
__int64 __fastcall sub_19D1320(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // r13
  __int64 result; // rax
  __int64 v8; // rdi
  unsigned int v9; // r13d
  __int64 v10; // r14
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rax
  unsigned int v22; // edx
  __int64 v23; // rdi
  int v24; // eax
  bool v25; // r14
  _QWORD *v26; // rbx
  unsigned int v27; // eax
  int v28; // eax
  bool v29; // r14
  _QWORD *v30; // rbx
  unsigned int v31; // eax
  __int64 v32; // [rsp+8h] [rbp-C8h]
  __int64 v33; // [rsp+10h] [rbp-C0h]
  __int64 v34; // [rsp+10h] [rbp-C0h]
  __int64 *v35; // [rsp+10h] [rbp-C0h]
  __int64 v36; // [rsp+10h] [rbp-C0h]
  __int64 *v37; // [rsp+10h] [rbp-C0h]
  __int64 v38; // [rsp+18h] [rbp-B8h]
  char v39; // [rsp+18h] [rbp-B8h]
  int v40; // [rsp+18h] [rbp-B8h]
  unsigned int v41; // [rsp+18h] [rbp-B8h]
  int v42; // [rsp+18h] [rbp-B8h]
  unsigned int v43; // [rsp+18h] [rbp-B8h]
  __m128i v44[3]; // [rsp+20h] [rbp-B0h] BYREF
  __m128i v45; // [rsp+50h] [rbp-80h] BYREF
  __int64 v46; // [rsp+60h] [rbp-70h]
  __int64 v47; // [rsp+68h] [rbp-68h]
  __int64 v48; // [rsp+70h] [rbp-60h]
  int v49; // [rsp+78h] [rbp-58h]
  __int64 v50; // [rsp+80h] [rbp-50h]
  __int64 v51; // [rsp+88h] [rbp-48h]

  v6 = sub_1649C60(*(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))));
  if ( v6 != sub_1649C60(*(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF))) )
    return 0;
  v8 = *(_QWORD *)(a3 + 24 * (3LL - (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)));
  v9 = *(_DWORD *)(v8 + 32);
  if ( v9 <= 0x40 )
  {
    if ( !*(_QWORD *)(v8 + 24) )
      goto LABEL_6;
    return 0;
  }
  if ( v9 != (unsigned int)sub_16A57B0(v8 + 24) )
    return 0;
LABEL_6:
  v10 = sub_1649C60(*(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))));
  v11 = *(_QWORD *)(a3 + 24 * (1LL - (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)));
  if ( v10 == sub_1649C60(v11) )
    return 0;
  v12 = *(_QWORD *)(a3 + 24 * (2LL - (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)));
  if ( *(_BYTE *)(v12 + 16) != 13 )
    v12 = 0;
  v13 = *(_QWORD *)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  if ( !v13 )
    BUG();
  if ( *(_BYTE *)(v13 + 16) != 13 || !v12 )
    return 0;
  v14 = *(_DWORD *)(v12 + 32) <= 0x40u ? *(_QWORD *)(v12 + 24) : **(_QWORD **)(v12 + 24);
  v15 = *(_DWORD *)(v13 + 32) <= 0x40u ? *(_QWORD *)(v13 + 24) : **(_QWORD **)(v13 + 24);
  if ( v15 > v14 )
    return 0;
  if ( !*(_QWORD *)(a1 + 32) )
    sub_4263D6(v11, a2, v14);
  v32 = (*(__int64 (__fastcall **)(__int64))(a1 + 40))(a1 + 16);
  v33 = *(_QWORD *)a1;
  v38 = *(_QWORD *)(a2 + 40);
  sub_141F730(&v45, a3);
  v16 = sub_141C340(v33, &v45, 0, (_QWORD *)(a2 + 24), v38, 0, 0, 0);
  if ( (v16 & 7) != 1 || a3 != (v16 & 0xFFFFFFFFFFFFFFF8LL) )
    return 0;
  sub_141F730(&v45, a3);
  sub_141F800(v44, a2);
  v39 = sub_134CB50(v32, (__int64)v44, (__int64)&v45);
  v17 = sub_16498A0(a2);
  v18 = *(_QWORD *)(a2 + 48);
  v46 = a2 + 24;
  v47 = v17;
  v19 = *(_QWORD *)(a2 + 40);
  v45.m128i_i64[0] = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v45.m128i_i64[1] = v19;
  v44[0].m128i_i64[0] = v18;
  if ( v18 )
  {
    sub_1623A60((__int64)v44, v18, 2);
    if ( v45.m128i_i64[0] )
      sub_161E7C0((__int64)&v45, v45.m128i_i64[0]);
    v45.m128i_i64[0] = v44[0].m128i_i64[0];
    if ( v44[0].m128i_i64[0] )
      sub_1623210((__int64)v44, (unsigned __int8 *)v44[0].m128i_i64[0], (__int64)&v45);
  }
  v20 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v21 = *(_QWORD *)(a2 + 24 * (3 - v20));
  v22 = *(_DWORD *)(v21 + 32);
  v23 = v21 + 24;
  if ( v39 )
  {
    if ( v22 <= 0x40 )
    {
      v25 = *(_QWORD *)(v21 + 24) == 0;
    }
    else
    {
      v34 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      v40 = *(_DWORD *)(v21 + 32);
      v24 = sub_16A57B0(v23);
      v20 = v34;
      v25 = v40 == v24;
    }
    v35 = *(__int64 **)(a2 + 24 * (2 - v20));
    v41 = sub_15603A0((_QWORD *)(a3 + 56), 1);
    v26 = *(_QWORD **)(a3 + 24 * (1LL - (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)));
    v27 = sub_15603A0((_QWORD *)(a2 + 56), 0);
    sub_15E7940(
      v45.m128i_i64,
      *(_QWORD **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)),
      v27,
      v26,
      v41,
      v35,
      !v25,
      0,
      0,
      0);
  }
  else
  {
    if ( v22 <= 0x40 )
    {
      v29 = *(_QWORD *)(v21 + 24) == 0;
    }
    else
    {
      v36 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      v42 = *(_DWORD *)(v21 + 32);
      v28 = sub_16A57B0(v23);
      v20 = v36;
      v29 = v42 == v28;
    }
    v37 = *(__int64 **)(a2 + 24 * (2 - v20));
    v43 = sub_15603A0((_QWORD *)(a3 + 56), 1);
    v30 = *(_QWORD **)(a3 + 24 * (1LL - (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)));
    v31 = sub_15603A0((_QWORD *)(a2 + 56), 0);
    sub_15E7430(
      v45.m128i_i64,
      *(_QWORD **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)),
      v31,
      v30,
      v43,
      v37,
      !v29,
      0,
      0,
      0,
      0);
  }
  sub_14191F0(*(_QWORD *)a1, a2);
  sub_15F20C0((_QWORD *)a2);
  result = 1;
  if ( v45.m128i_i64[0] )
  {
    sub_161E7C0((__int64)&v45, v45.m128i_i64[0]);
    return 1;
  }
  return result;
}
