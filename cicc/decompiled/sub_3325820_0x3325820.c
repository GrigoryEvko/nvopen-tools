// Function: sub_3325820
// Address: 0x3325820
//
__int64 __fastcall sub_3325820(__int64 a1, int *a2, int a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v9; // rdi
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // r15
  char v13; // al
  int v14; // eax
  __int64 v15; // rdi
  int v16; // esi
  int v17; // edx
  int v18; // r9d
  __int64 v19; // rdx
  __int64 v21; // [rsp-50h] [rbp-F0h]
  __int64 v22; // [rsp-48h] [rbp-E8h]
  __int128 v23; // [rsp-40h] [rbp-E0h]
  __int64 v24; // [rsp-30h] [rbp-D0h]
  __int128 v25; // [rsp-10h] [rbp-B0h]
  int v26; // [rsp+0h] [rbp-A0h]
  int v27; // [rsp+8h] [rbp-98h]
  __m128i v28; // [rsp+20h] [rbp-80h]
  __int128 v29; // [rsp+30h] [rbp-70h]
  __int64 v30; // [rsp+40h] [rbp-60h]
  __int64 v31; // [rsp+50h] [rbp-50h] BYREF
  __int64 v32; // [rsp+58h] [rbp-48h]
  __int64 v33; // [rsp+60h] [rbp-40h]
  __int64 v34; // [rsp+68h] [rbp-38h]

  v9 = *(_QWORD *)(a1 + 16);
  if ( *((_QWORD *)a2 + 2) )
  {
    v27 = a5;
    v10 = *((_QWORD *)a2 + 3);
    v11 = *((_QWORD *)a2 + 10);
    v26 = a4;
    v12 = *((_QWORD *)a2 + 2);
    v28 = _mm_loadu_si128((const __m128i *)a2 + 3);
    v29 = (__int128)_mm_loadu_si128((const __m128i *)a2 + 4);
    v31 = 0;
    v32 = 0;
    v33 = 0;
    v34 = 0;
    v30 = v11;
    v13 = sub_33CC4A0(v9, 5, 0);
    v14 = sub_33F5040(
            v9,
            v12,
            v10,
            a3,
            v26,
            v27,
            v28.m128i_i64[0],
            v28.m128i_i64[1],
            v29,
            v30,
            5,
            0,
            v13,
            0,
            (__int64)&v31);
    v15 = *(_QWORD *)(a1 + 16);
    v16 = *a2;
    v18 = v17;
    v19 = *((_QWORD *)a2 + 1);
    v24 = *((_QWORD *)a2 + 13);
    v23 = *(_OWORD *)(a2 + 22);
    v22 = *((_QWORD *)a2 + 5);
    v21 = *((_QWORD *)a2 + 4);
    v31 = 0;
    v32 = 0;
    v33 = 0;
    v34 = 0;
    return sub_33F1F00(v15, v16, v19, a3, v14, v18, v21, v22, v23, v24, 0, 0, (__int64)&v31, 0);
  }
  else
  {
    *((_QWORD *)&v25 + 1) = a5;
    *(_QWORD *)&v25 = a4;
    return sub_33FAF80(v9, 234, a3, *a2, *((_QWORD *)a2 + 1), a6, v25);
  }
}
