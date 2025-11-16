// Function: sub_378EED0
// Address: 0x378eed0
//
__int64 __fastcall sub_378EED0(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 (__fastcall *v4)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v5; // rax
  unsigned __int16 v6; // si
  __int64 v7; // r8
  __int64 v8; // rax
  unsigned int v9; // r13d
  __int64 v10; // r15
  int v11; // r9d
  __int64 v12; // rsi
  _QWORD *v13; // r9
  __int128 *v14; // r12
  unsigned int v15; // esi
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // r12
  __int64 v20; // rsi
  __int64 v21; // r12
  __int64 v22; // rsi
  unsigned __int8 *v23; // rax
  __int64 v24; // rdx
  _QWORD *v25; // [rsp+8h] [rbp-58h]
  __int64 v26; // [rsp+10h] [rbp-50h] BYREF
  int v27; // [rsp+18h] [rbp-48h]
  __int64 v28; // [rsp+20h] [rbp-40h]

  v4 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v7 = *((_QWORD *)v5 + 1);
  v8 = a1[1];
  if ( v4 == sub_2D56A50 )
  {
    HIWORD(v9) = 0;
    sub_2FE6CC0((__int64)&v26, *a1, *(_QWORD *)(v8 + 64), v6, v7);
    LOWORD(v9) = v27;
    v10 = v28;
  }
  else
  {
    v9 = v4(*a1, *(_QWORD *)(v8 + 64), v6, v7);
    v10 = v24;
  }
  if ( sub_33CB110(*(_DWORD *)(a2 + 24)) )
  {
    v12 = *(_QWORD *)(a2 + 80);
    v13 = (_QWORD *)a1[1];
    v14 = *(__int128 **)(a2 + 40);
    v26 = v12;
    if ( v12 )
    {
      v25 = v13;
      sub_B96E90((__int64)&v26, v12, 1);
      v13 = v25;
    }
    v15 = *(_DWORD *)(a2 + 24);
    v27 = *(_DWORD *)(a2 + 72);
    v16 = sub_340F900(v13, v15, (__int64)&v26, v9, v10, (__int64)v13, *v14, *(__int128 *)((char *)v14 + 40), v14[5]);
    v17 = v26;
    v18 = v16;
    if ( v26 )
LABEL_7:
      sub_B91220((__int64)&v26, v17);
  }
  else
  {
    v20 = *(_QWORD *)(a2 + 80);
    v21 = a1[1];
    v26 = v20;
    if ( v20 )
      sub_B96E90((__int64)&v26, v20, 1);
    v22 = *(unsigned int *)(a2 + 24);
    v27 = *(_DWORD *)(a2 + 72);
    v23 = sub_33FAF80(v21, v22, (__int64)&v26, v9, v10, v11, a3);
    v17 = v26;
    v18 = (__int64)v23;
    if ( v26 )
      goto LABEL_7;
  }
  return v18;
}
