// Function: sub_382A8A0
// Address: 0x382a8a0
//
__int64 __fastcall sub_382A8A0(__int64 *a1, __int64 a2, __m128i a3)
{
  unsigned int v3; // r14d
  __int64 v5; // rsi
  __int64 v6; // r9
  __int64 v7; // rdx
  __int16 *v8; // rax
  __int64 v9; // r10
  unsigned __int16 v10; // si
  __int64 v11; // r8
  __int64 (__fastcall *v12)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // rcx
  __int64 v16; // r8
  int v17; // r9d
  unsigned int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // r8
  unsigned __int8 *v21; // r14
  __int64 v22; // rdx
  __int64 v23; // r15
  __int64 v24; // r9
  __int64 v25; // r14
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int128 v29; // [rsp-30h] [rbp-90h]
  __int64 v30; // [rsp+0h] [rbp-60h] BYREF
  int v31; // [rsp+8h] [rbp-58h]
  unsigned int v32; // [rsp+10h] [rbp-50h] BYREF
  __int64 v33; // [rsp+18h] [rbp-48h]
  __int64 v34; // [rsp+20h] [rbp-40h]

  v5 = *(_QWORD *)(a2 + 80);
  v30 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v30, v5, 1);
  v6 = *a1;
  v7 = a1[1];
  v31 = *(_DWORD *)(a2 + 72);
  v8 = *(__int16 **)(a2 + 48);
  v9 = *(_QWORD *)(v7 + 64);
  v10 = *v8;
  v11 = *((_QWORD *)v8 + 1);
  v12 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v6 + 592LL);
  if ( v12 == sub_2D56A50 )
  {
    v13 = v10;
    v14 = v6;
    sub_2FE6CC0((__int64)&v32, v6, *(_QWORD *)(v7 + 64), v13, v11);
    LOWORD(v18) = v33;
    v19 = v34;
    LOWORD(v32) = v33;
    v33 = v34;
  }
  else
  {
    v28 = v10;
    v14 = v9;
    v18 = v12(v6, v9, v28, v11);
    v32 = v18;
    v33 = v19;
  }
  if ( (_WORD)v18 )
  {
    v20 = 0;
    LOWORD(v18) = word_4456580[(unsigned __int16)v18 - 1];
  }
  else
  {
    v18 = sub_3009970((__int64)&v32, v14, v19, v15, v16);
    HIWORD(v3) = HIWORD(v18);
    v20 = v27;
  }
  LOWORD(v3) = v18;
  v21 = sub_33FAF80(a1[1], 215, (__int64)&v30, v3, v20, v17, a3);
  v23 = v22;
  if ( sub_33CB110(*(_DWORD *)(a2 + 24)) )
  {
    *((_QWORD *)&v29 + 1) = v23;
    *(_QWORD *)&v29 = v21;
    v25 = sub_340F900(
            (_QWORD *)a1[1],
            *(_DWORD *)(a2 + 24),
            (__int64)&v30,
            v32,
            v33,
            v24,
            v29,
            *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
            *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
  }
  else
  {
    v25 = (__int64)sub_33FAF80(a1[1], *(unsigned int *)(a2 + 24), (__int64)&v30, v32, v33, v24, a3);
  }
  if ( v30 )
    sub_B91220((__int64)&v30, v30);
  return v25;
}
