// Function: sub_382E080
// Address: 0x382e080
//
__int64 __fastcall sub_382E080(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 (__fastcall *v4)(__int64, __int64, unsigned int, __int64); // r9
  __int16 *v5; // rax
  unsigned __int16 v6; // si
  __int64 v7; // r8
  __int64 v8; // rax
  unsigned int v9; // r15d
  __int16 v10; // r13
  __int64 v11; // rsi
  __int64 v12; // rsi
  unsigned __int16 *v13; // rax
  __int64 v14; // r9
  int v15; // edx
  _QWORD *v16; // rdi
  __int64 v17; // rsi
  __int64 v18; // r14
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r8
  __int64 v23; // rax
  __int64 v24; // rax
  int v25; // eax
  unsigned __int16 *v26; // rax
  int v27; // eax
  __int64 v28; // rdx
  _QWORD *v29; // r12
  unsigned __int16 *v30; // rax
  __int128 v31; // rax
  __int128 v32; // [rsp+0h] [rbp-80h]
  __int64 v33; // [rsp+18h] [rbp-68h]
  __int64 v34; // [rsp+20h] [rbp-60h] BYREF
  int v35; // [rsp+28h] [rbp-58h]
  _BYTE v36[8]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v37; // [rsp+38h] [rbp-48h]
  __int64 v38; // [rsp+40h] [rbp-40h]

  v4 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v7 = *((_QWORD *)v5 + 1);
  v8 = a1[1];
  if ( v4 == sub_2D56A50 )
  {
    HIWORD(v9) = 0;
    sub_2FE6CC0((__int64)v36, *a1, *(_QWORD *)(v8 + 64), v6, v7);
    v10 = v37;
    v33 = v38;
  }
  else
  {
    v27 = v4(*a1, *(_QWORD *)(v8 + 64), v6, v7);
    v33 = v28;
    HIWORD(v9) = HIWORD(v27);
    v10 = v27;
  }
  v11 = *(_QWORD *)(a2 + 80);
  v34 = v11;
  if ( v11 )
    sub_B96E90((__int64)&v34, v11, 1);
  v12 = *a1;
  v35 = *(_DWORD *)(a2 + 72);
  v13 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL)
                           + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL));
  sub_2FE6CC0((__int64)v36, v12, *(_QWORD *)(a1[1] + 64), *v13, *((_QWORD *)v13 + 1));
  if ( v36[0] != 1
    || (v20 = sub_37AE0F0((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL)),
        v14 = v21,
        v22 = v20,
        v23 = *(_QWORD *)(v20 + 48) + 16LL * (unsigned int)v21,
        *(_WORD *)v23 != v10) )
  {
    v15 = *(_DWORD *)(a2 + 64);
LABEL_7:
    v16 = (_QWORD *)a1[1];
    v17 = *(unsigned int *)(a2 + 24);
    LOWORD(v9) = v10;
    if ( v15 == 1 )
      v18 = (__int64)sub_33FAF80((__int64)v16, v17, (__int64)&v34, v9, v33, v14, a3);
    else
      v18 = sub_340F900(
              v16,
              v17,
              (__int64)&v34,
              v9,
              v33,
              v14,
              *(_OWORD *)*(_QWORD *)(a2 + 40),
              *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
              *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
    goto LABEL_9;
  }
  v24 = *(_QWORD *)(v23 + 8);
  v15 = *(_DWORD *)(a2 + 64);
  if ( !v10 && v33 != v24 || v15 != 1 )
    goto LABEL_7;
  v25 = *(_DWORD *)(a2 + 24);
  if ( v25 == 213 )
  {
    v29 = (_QWORD *)a1[1];
    *(_QWORD *)&v32 = v22;
    *((_QWORD *)&v32 + 1) = v14;
    LOWORD(v9) = v10;
    v30 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL));
    *(_QWORD *)&v31 = sub_33F7D60(v29, *v30, *((_QWORD *)v30 + 1));
    v18 = (__int64)sub_3406EB0(v29, 0xDEu, (__int64)&v34, v9, v33, *((__int64 *)&v32 + 1), v32, v31);
  }
  else
  {
    v18 = v22;
    if ( v25 == 214 )
    {
      v26 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL)
                               + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL));
      v18 = (__int64)sub_34070B0((_QWORD *)a1[1], v22, v14, (__int64)&v34, *v26, *((_QWORD *)v26 + 1), a3);
    }
  }
LABEL_9:
  if ( v34 )
    sub_B91220((__int64)&v34, v34);
  return v18;
}
