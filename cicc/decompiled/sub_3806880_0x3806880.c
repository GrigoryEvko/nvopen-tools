// Function: sub_3806880
// Address: 0x3806880
//
unsigned __int8 *__fastcall sub_3806880(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 (__fastcall *v4)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v5; // rax
  unsigned __int16 v6; // si
  __int64 v7; // r8
  __int64 v8; // rax
  unsigned int v9; // eax
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned int v12; // eax
  unsigned int v13; // r13d
  __int64 v14; // r12
  unsigned __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // r12
  unsigned __int8 *v18; // r12
  __int64 v19; // rdx
  __int64 v20; // r13
  unsigned __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rsi
  _QWORD *v24; // r14
  unsigned __int64 v25; // r8
  __int64 v26; // r9
  unsigned __int8 *v27; // r12
  __int64 v29; // rdx
  __int128 v30; // [rsp-20h] [rbp-B0h]
  __int128 v31; // [rsp-10h] [rbp-A0h]
  unsigned __int64 v32; // [rsp+0h] [rbp-90h]
  __int64 v33; // [rsp+8h] [rbp-88h]
  unsigned int v34; // [rsp+10h] [rbp-80h] BYREF
  __int64 v35; // [rsp+18h] [rbp-78h]
  unsigned __int64 v36; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v37; // [rsp+28h] [rbp-68h]
  __int64 v38; // [rsp+30h] [rbp-60h] BYREF
  int v39; // [rsp+38h] [rbp-58h]
  __int64 v40; // [rsp+40h] [rbp-50h] BYREF
  __int64 v41; // [rsp+48h] [rbp-48h]
  __int64 v42; // [rsp+50h] [rbp-40h]

  v4 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v7 = *((_QWORD *)v5 + 1);
  v8 = a1[1];
  if ( v4 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v40, *a1, *(_QWORD *)(v8 + 64), v6, v7);
    LOWORD(v9) = v41;
    LOWORD(v34) = v41;
    v35 = v42;
  }
  else
  {
    v9 = v4(*a1, *(_QWORD *)(v8 + 64), v6, v7);
    v34 = v9;
    v35 = v29;
  }
  if ( (_WORD)v9 )
  {
    if ( (_WORD)v9 == 1 || (unsigned __int16)(v9 - 504) <= 7u )
      BUG();
    v11 = 16LL * ((unsigned __int16)v9 - 1);
    v10 = *(_QWORD *)&byte_444C4A0[v11];
    LOBYTE(v11) = byte_444C4A0[v11 + 8];
  }
  else
  {
    v10 = sub_3007260((__int64)&v34);
    v40 = v10;
    v41 = v11;
  }
  v38 = v10;
  LOBYTE(v39) = v11;
  v12 = sub_CA1930(&v38);
  v13 = v12 - 1;
  v37 = v12;
  v14 = ~(1LL << ((unsigned __int8)v12 - 1));
  if ( v12 <= 0x40 )
  {
    v15 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v12;
    if ( !v12 )
      v15 = 0;
    v36 = v15;
    goto LABEL_9;
  }
  sub_C43690((__int64)&v36, -1, 1);
  if ( v37 <= 0x40 )
  {
LABEL_9:
    v36 &= v14;
    goto LABEL_10;
  }
  *(_QWORD *)(v36 + 8LL * (v13 >> 6)) &= v14;
LABEL_10:
  v16 = *(_QWORD *)(a2 + 80);
  v17 = a1[1];
  v38 = v16;
  if ( v16 )
    sub_B96E90((__int64)&v38, v16, 1);
  v39 = *(_DWORD *)(a2 + 72);
  v18 = sub_34007B0(v17, (__int64)&v36, (__int64)&v38, v34, v35, 0, a3, 0);
  v20 = v19;
  if ( v38 )
    sub_B91220((__int64)&v38, v38);
  v21 = sub_3805E70((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v23 = *(_QWORD *)(a2 + 80);
  v24 = (_QWORD *)a1[1];
  v25 = v21;
  v26 = v22;
  v38 = v23;
  if ( v23 )
  {
    v33 = v22;
    v32 = v21;
    sub_B96E90((__int64)&v38, v23, 1);
    v25 = v32;
    v26 = v33;
  }
  *((_QWORD *)&v31 + 1) = v20;
  *(_QWORD *)&v31 = v18;
  *((_QWORD *)&v30 + 1) = v26;
  *(_QWORD *)&v30 = v25;
  v39 = *(_DWORD *)(a2 + 72);
  v27 = sub_3406EB0(v24, 0xBAu, (__int64)&v38, v34, v35, v26, v30, v31);
  if ( v38 )
    sub_B91220((__int64)&v38, v38);
  if ( v37 > 0x40 && v36 )
    j_j___libc_free_0_0(v36);
  return v27;
}
