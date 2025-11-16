// Function: sub_3790EE0
// Address: 0x3790ee0
//
unsigned __int8 *__fastcall sub_3790EE0(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 (__fastcall *v4)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v5; // rax
  unsigned __int16 v6; // si
  __int64 v7; // r8
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // rcx
  unsigned __int8 *v11; // rax
  __int64 v12; // rsi
  _QWORD *v13; // r13
  __int64 v14; // rdx
  __int64 v15; // r15
  __int64 v16; // r9
  unsigned __int8 *v17; // r14
  unsigned __int16 *v18; // rdx
  __int64 v19; // r8
  __int64 v20; // rcx
  unsigned __int8 *v21; // r14
  __int64 v23; // rdx
  __int128 v24; // [rsp-20h] [rbp-90h]
  __int64 v25; // [rsp+8h] [rbp-68h]
  __int64 v26; // [rsp+10h] [rbp-60h]
  __int64 v27; // [rsp+18h] [rbp-58h]
  __int64 v28; // [rsp+20h] [rbp-50h] BYREF
  int v29; // [rsp+28h] [rbp-48h]
  __int64 v30; // [rsp+30h] [rbp-40h]

  v4 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v7 = *((_QWORD *)v5 + 1);
  v8 = a1[1];
  if ( v4 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v28, *a1, *(_QWORD *)(v8 + 64), v6, v7);
    v9 = v30;
    v10 = (unsigned __int16)v29;
  }
  else
  {
    v10 = v4(*a1, *(_QWORD *)(v8 + 64), v6, v7);
    v9 = v23;
  }
  v11 = sub_3790540((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), v10, v9, 1, a3);
  v12 = *(_QWORD *)(a2 + 80);
  v13 = (_QWORD *)a1[1];
  v15 = v14;
  v16 = *(_QWORD *)(a2 + 40);
  v17 = v11;
  v18 = (unsigned __int16 *)(*((_QWORD *)v11 + 6) + 16LL * (unsigned int)v14);
  v19 = *((_QWORD *)v18 + 1);
  v20 = *v18;
  v28 = v12;
  if ( v12 )
  {
    v25 = v20;
    v26 = v16;
    v27 = v19;
    sub_B96E90((__int64)&v28, v12, 1);
    v20 = v25;
    v16 = v26;
    v19 = v27;
  }
  v29 = *(_DWORD *)(a2 + 72);
  *((_QWORD *)&v24 + 1) = v15;
  *(_QWORD *)&v24 = v17;
  v21 = sub_3406EB0(v13, 4u, (__int64)&v28, v20, v19, v16, v24, *(_OWORD *)(v16 + 40));
  if ( v28 )
    sub_B91220((__int64)&v28, v28);
  return v21;
}
