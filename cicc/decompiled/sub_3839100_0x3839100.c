// Function: sub_3839100
// Address: 0x3839100
//
unsigned __int8 *__fastcall sub_3839100(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r13
  __int64 v7; // r11
  __int64 v8; // rsi
  _QWORD *v9; // r9
  unsigned __int16 *v10; // rax
  __int64 v11; // r15
  unsigned int v12; // r12d
  unsigned int v13; // esi
  unsigned __int8 *v14; // rax
  __int64 v15; // rsi
  unsigned __int8 *v16; // r12
  __m128i v18; // xmm0
  __int64 v19; // r9
  __int64 v20; // r11
  __int64 v21; // rsi
  _QWORD *v22; // r9
  unsigned __int16 *v23; // rax
  __int64 v24; // r15
  unsigned int v25; // r12d
  __int64 v26; // rsi
  unsigned __int8 *v27; // rax
  unsigned __int8 *v28; // rax
  unsigned int v29; // edx
  unsigned __int8 *v30; // rax
  unsigned int v31; // edx
  __int128 v32; // [rsp-30h] [rbp-D0h]
  __int128 v33; // [rsp-10h] [rbp-B0h]
  unsigned __int64 v34; // [rsp+0h] [rbp-A0h]
  __int64 v35; // [rsp+8h] [rbp-98h]
  __int64 v36; // [rsp+8h] [rbp-98h]
  __int128 v37; // [rsp+10h] [rbp-90h]
  unsigned __int64 v38; // [rsp+20h] [rbp-80h]
  __int64 v39; // [rsp+28h] [rbp-78h]
  __int64 v40; // [rsp+28h] [rbp-78h]
  __int128 v41; // [rsp+30h] [rbp-70h]
  unsigned int v42; // [rsp+40h] [rbp-60h]
  _QWORD *v43; // [rsp+40h] [rbp-60h]
  _QWORD *v44; // [rsp+40h] [rbp-60h]
  unsigned __int8 *v45; // [rsp+48h] [rbp-58h]
  __int64 v46; // [rsp+50h] [rbp-50h] BYREF
  int v47; // [rsp+58h] [rbp-48h]

  *(_QWORD *)&v41 = sub_37AE0F0((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v4 = *(_QWORD *)(a2 + 40);
  *((_QWORD *)&v41 + 1) = v5;
  v42 = v5;
  v6 = *(unsigned int *)(v4 + 48);
  v45 = *(unsigned __int8 **)(v4 + 40);
  if ( *(_DWORD *)(a2 + 24) == 402 )
  {
    v18 = _mm_loadu_si128((const __m128i *)(v4 + 80));
    v34 = *(_QWORD *)(v4 + 40);
    v35 = *(_QWORD *)(v4 + 48);
    v37 = (__int128)_mm_loadu_si128((const __m128i *)(v4 + 120));
    sub_2FE6CC0(
      (__int64)&v46,
      *a1,
      *(_QWORD *)(a1[1] + 64),
      *(unsigned __int16 *)(*((_QWORD *)v45 + 6) + 16 * v6),
      *(_QWORD *)(*((_QWORD *)v45 + 6) + 16 * v6 + 8));
    v20 = v35;
    if ( (_BYTE)v46 == 1 )
    {
      v30 = sub_3838540((__int64)a1, v34, v35, v18.m128i_i64[0], v18.m128i_i64[1], v18, v19, v37);
      v20 = v35;
      v45 = v30;
      v6 = v31;
    }
    v21 = *(_QWORD *)(a2 + 80);
    v22 = (_QWORD *)a1[1];
    v23 = (unsigned __int16 *)(*(_QWORD *)(v41 + 48) + 16LL * v42);
    v24 = *((_QWORD *)v23 + 1);
    v25 = *v23;
    v46 = v21;
    if ( v21 )
    {
      v44 = v22;
      v36 = v20;
      sub_B96E90((__int64)&v46, v21, 1);
      v20 = v36;
      v22 = v44;
    }
    v26 = *(unsigned int *)(a2 + 24);
    v47 = *(_DWORD *)(a2 + 72);
    *((_QWORD *)&v32 + 1) = v6 | v20 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v32 = v45;
    v27 = sub_33FC130(v22, v26, (__int64)&v46, v25, v24, (__int64)v22, v41, v32, *(_OWORD *)&v18, v37);
    v15 = v46;
    v16 = v27;
    if ( v46 )
      goto LABEL_7;
  }
  else
  {
    v38 = *(_QWORD *)(v4 + 40);
    v39 = *(_QWORD *)(v4 + 48);
    sub_2FE6CC0(
      (__int64)&v46,
      *a1,
      *(_QWORD *)(a1[1] + 64),
      *(unsigned __int16 *)(*(_QWORD *)(v38 + 48) + 16 * v6),
      *(_QWORD *)(*(_QWORD *)(v38 + 48) + 16 * v6 + 8));
    v7 = v39;
    if ( (_BYTE)v46 == 1 )
    {
      v28 = sub_37AF270((__int64)a1, v38, v39, a3);
      v7 = v39;
      v45 = v28;
      v6 = v29;
    }
    v8 = *(_QWORD *)(a2 + 80);
    v9 = (_QWORD *)a1[1];
    v10 = (unsigned __int16 *)(*(_QWORD *)(v41 + 48) + 16LL * v42);
    v11 = *((_QWORD *)v10 + 1);
    v12 = *v10;
    v46 = v8;
    if ( v8 )
    {
      v40 = v7;
      v43 = v9;
      sub_B96E90((__int64)&v46, v8, 1);
      v7 = v40;
      v9 = v43;
    }
    v13 = *(_DWORD *)(a2 + 24);
    v47 = *(_DWORD *)(a2 + 72);
    *((_QWORD *)&v33 + 1) = v6 | v7 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v33 = v45;
    v14 = sub_3406EB0(v9, v13, (__int64)&v46, v12, v11, (__int64)v9, v41, v33);
    v15 = v46;
    v16 = v14;
    if ( v46 )
LABEL_7:
      sub_B91220((__int64)&v46, v15);
  }
  return v16;
}
