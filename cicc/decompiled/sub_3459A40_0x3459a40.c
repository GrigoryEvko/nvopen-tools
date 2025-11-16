// Function: sub_3459A40
// Address: 0x3459a40
//
__int64 __fastcall sub_3459A40(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  __int128 v7; // xmm0
  __int64 v8; // r14
  __int64 v9; // r15
  unsigned __int16 *v10; // rax
  unsigned int v11; // ebx
  __int128 v12; // rax
  __int64 v13; // r9
  __int128 v14; // rax
  __int128 v15; // rax
  __int64 v16; // r9
  __int128 v17; // rax
  __int64 v18; // r9
  __int128 v19; // rax
  __int64 v20; // r9
  __int64 v21; // r14
  __int128 v23; // [rsp-20h] [rbp-A0h]
  __int128 v24; // [rsp-10h] [rbp-90h]
  __int128 v25; // [rsp-10h] [rbp-90h]
  __int128 v26; // [rsp-10h] [rbp-90h]
  __int64 v27; // [rsp+8h] [rbp-78h]
  __int64 v28; // [rsp+10h] [rbp-70h]
  __int128 v29; // [rsp+10h] [rbp-70h]
  __int128 v30; // [rsp+30h] [rbp-50h]
  __int64 v31; // [rsp+40h] [rbp-40h] BYREF
  int v32; // [rsp+48h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 40);
  v6 = *(_QWORD *)(a2 + 80);
  v7 = (__int128)_mm_loadu_si128((const __m128i *)v5);
  v31 = v6;
  v8 = *(_QWORD *)(v5 + 80);
  v9 = *(_QWORD *)(v5 + 88);
  v30 = (__int128)_mm_loadu_si128((const __m128i *)(v5 + 40));
  if ( v6 )
    sub_B96E90((__int64)&v31, v6, 1);
  v32 = *(_DWORD *)(a2 + 72);
  v10 = *(unsigned __int16 **)(a2 + 48);
  v11 = *v10;
  v28 = *((_QWORD *)v10 + 1);
  *(_QWORD *)&v12 = sub_34015B0((__int64)a3, (__int64)&v31, *v10, v28, 0, 0, (__m128i)v7);
  *((_QWORD *)&v24 + 1) = v9;
  *(_QWORD *)&v24 = v8;
  v27 = v28;
  *(_QWORD *)&v14 = sub_33FC130(a3, 407, (__int64)&v31, v11, v28, v13, v7, v12, v30, v24);
  v29 = v14;
  *(_QWORD *)&v15 = sub_3400BD0((__int64)a3, 1, (__int64)&v31, v11, v27, 0, (__m128i)v7, 0);
  *((_QWORD *)&v23 + 1) = v9;
  *(_QWORD *)&v23 = v8;
  *(_QWORD *)&v17 = sub_33FC130(a3, 404, (__int64)&v31, v11, v27, v16, v7, v15, v30, v23);
  *((_QWORD *)&v25 + 1) = v9;
  *(_QWORD *)&v25 = v8;
  *(_QWORD *)&v19 = sub_33FC130(a3, 396, (__int64)&v31, v11, v27, v18, v29, v17, v30, v25);
  *((_QWORD *)&v26 + 1) = v9;
  *(_QWORD *)&v26 = v8;
  v21 = sub_340F900(a3, 0x19Fu, (__int64)&v31, v11, v27, v20, v19, v30, v26);
  if ( v31 )
    sub_B91220((__int64)&v31, v31);
  return v21;
}
