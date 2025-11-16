// Function: sub_379BBB0
// Address: 0x379bbb0
//
unsigned __int8 *__fastcall sub_379BBB0(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 v4; // rsi
  __int64 v5; // r9
  __int64 (__fastcall *v6)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v7; // rax
  unsigned __int16 v8; // si
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // rdx
  __int64 v13; // r13
  __int128 v14; // rax
  __int128 v15; // kr00_16
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned int v18; // edx
  unsigned int v19; // ecx
  __int64 v20; // rdx
  _QWORD *v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // rax
  int v24; // r9d
  unsigned __int8 *v25; // r12
  __int64 v27; // rdx
  __int128 v28; // [rsp-20h] [rbp-F0h]
  __int64 v29; // [rsp+10h] [rbp-C0h]
  __int128 v30; // [rsp+10h] [rbp-C0h]
  __int64 v31; // [rsp+40h] [rbp-90h] BYREF
  int v32; // [rsp+48h] [rbp-88h]
  unsigned int v33; // [rsp+50h] [rbp-80h] BYREF
  __int64 v34; // [rsp+58h] [rbp-78h]
  __int64 v35; // [rsp+60h] [rbp-70h] BYREF
  __int64 v36; // [rsp+68h] [rbp-68h]
  __int128 v37; // [rsp+70h] [rbp-60h]
  __int64 v38; // [rsp+80h] [rbp-50h]
  unsigned __int64 v39; // [rsp+88h] [rbp-48h]
  __m128i v40; // [rsp+90h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 80);
  v31 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v31, v4, 1);
  v5 = *a1;
  v32 = *(_DWORD *)(a2 + 72);
  v6 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v5 + 592LL);
  v7 = *(__int16 **)(a2 + 48);
  v8 = *v7;
  v9 = *((_QWORD *)v7 + 1);
  v10 = a1[1];
  if ( v6 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v35, v5, *(_QWORD *)(v10 + 64), v8, v9);
    LOWORD(v33) = v36;
    v34 = v37;
  }
  else
  {
    v33 = v6(v5, *(_QWORD *)(v10 + 64), v8, v9);
    v34 = v27;
  }
  v11 = sub_379AB60((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v13 = v12;
  *(_QWORD *)&v14 = sub_379AB60(
                      (__int64)a1,
                      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v15 = v14;
  if ( *(_DWORD *)(a2 + 64) == 2 )
  {
    *((_QWORD *)&v28 + 1) = v13;
    *(_QWORD *)&v28 = v11;
    v25 = sub_3405C90(
            (_QWORD *)a1[1],
            *(_DWORD *)(a2 + 24),
            (__int64)&v31,
            v33,
            v34,
            *(_DWORD *)(a2 + 28),
            a3,
            v28,
            v14);
  }
  else
  {
    if ( !(_WORD)v33 )
    {
      v30 = v14;
      sub_3007240((__int64)&v33);
      v15 = v30;
    }
    v16 = *(_QWORD *)(a2 + 40);
    v29 = *(_QWORD *)(v16 + 88);
    v17 = sub_379AB60((__int64)a1, *(_QWORD *)(v16 + 80), v29);
    v19 = v18;
    v20 = v17;
    v35 = v11;
    v21 = (_QWORD *)a1[1];
    v36 = v13;
    v22 = *(unsigned int *)(a2 + 24);
    v23 = *(_QWORD *)(a2 + 40);
    v37 = v15;
    v38 = v20;
    v24 = *(_DWORD *)(a2 + 28);
    v39 = v19 | v29 & 0xFFFFFFFF00000000LL;
    v40 = _mm_loadu_si128((const __m128i *)(v23 + 120));
    v25 = sub_33FBA10(v21, v22, (__int64)&v31, v33, v34, v24, (__int64)&v35, 4);
  }
  if ( v31 )
    sub_B91220((__int64)&v31, v31);
  return v25;
}
