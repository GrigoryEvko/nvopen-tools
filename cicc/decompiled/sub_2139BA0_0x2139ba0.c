// Function: sub_2139BA0
// Address: 0x2139ba0
//
__int64 *__fastcall sub_2139BA0(__int64 a1, unsigned __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 *v7; // r14
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // r15
  __int64 *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 *v13; // r8
  __int64 v14; // r9
  unsigned __int8 *v15; // rax
  unsigned int v16; // r10d
  __int64 v17; // rcx
  const void **v18; // r11
  __int64 *v19; // rdi
  __int64 v20; // rsi
  __int64 *v21; // r14
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // r15
  __int64 *v24; // r9
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  unsigned int v29; // edx
  const __m128i *v30; // r9
  __int128 v32; // [rsp-20h] [rbp-B0h]
  __int128 v33; // [rsp-10h] [rbp-A0h]
  __int64 v34; // [rsp+8h] [rbp-88h]
  const void **v35; // [rsp+8h] [rbp-88h]
  __int64 *v36; // [rsp+10h] [rbp-80h]
  __int64 v37; // [rsp+10h] [rbp-80h]
  __int64 v38; // [rsp+18h] [rbp-78h]
  unsigned int v39; // [rsp+20h] [rbp-70h]
  unsigned __int64 v40; // [rsp+20h] [rbp-70h]
  const void **v41; // [rsp+28h] [rbp-68h]
  unsigned int v42; // [rsp+28h] [rbp-68h]
  __int64 *v43; // [rsp+28h] [rbp-68h]
  __int64 v44; // [rsp+30h] [rbp-60h]
  unsigned __int64 v45; // [rsp+30h] [rbp-60h]
  __int16 *v46; // [rsp+38h] [rbp-58h]
  __int64 *v47; // [rsp+40h] [rbp-50h]
  __int64 v48; // [rsp+50h] [rbp-40h] BYREF
  int v49; // [rsp+58h] [rbp-38h]

  v7 = sub_2139210(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), a3, a4, a5);
  v9 = v8;
  v10 = sub_2139210(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL), a3, a4, a5);
  v12 = *(_QWORD *)(a2 + 72);
  v13 = v10;
  v14 = v11;
  v15 = (unsigned __int8 *)(*(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL)
                          + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL));
  v16 = *v15;
  v17 = *(unsigned __int8 *)(v7[5] + 16LL * (unsigned int)v9);
  v18 = *(const void ***)(v7[5] + 16LL * (unsigned int)v9 + 8);
  v44 = *((_QWORD *)v15 + 1);
  v48 = v12;
  if ( v12 )
  {
    v34 = v17;
    v39 = v16;
    v36 = v13;
    v38 = v11;
    v41 = v18;
    sub_1623A60((__int64)&v48, v12, 2);
    v17 = v34;
    v16 = v39;
    v13 = v36;
    v14 = v38;
    v18 = v41;
  }
  *((_QWORD *)&v33 + 1) = v14;
  *(_QWORD *)&v33 = v13;
  v19 = *(__int64 **)(a1 + 8);
  v20 = (unsigned int)(*(_WORD *)(a2 + 24) != 71) + 52;
  v42 = v16;
  v49 = *(_DWORD *)(a2 + 64);
  v21 = sub_1D332F0(v19, v20, (__int64)&v48, v17, v18, 0, *(double *)a3.m128i_i64, a4, a5, (__int64)v7, v9, v33);
  v23 = v22;
  v24 = sub_1D3BC50(*(__int64 **)(a1 + 8), (__int64)v21, v22, (__int64)&v48, v42, v44, a3, a4, a5);
  v25 = *(_QWORD *)(a2 + 40);
  v46 = (__int16 *)v26;
  v45 = (unsigned __int64)v24;
  v43 = *(__int64 **)(a1 + 8);
  v37 = v26;
  v35 = *(const void ***)(v25 + 24);
  v40 = *(unsigned __int8 *)(v25 + 16);
  v27 = sub_1D28D50(v43, 0x16u, v26, v40, (__int64)v35, (__int64)v24);
  *((_QWORD *)&v32 + 1) = v23;
  *(_QWORD *)&v32 = v21;
  v47 = sub_1D3A900(v43, 0x89u, (__int64)&v48, v40, v35, 0, (__m128)a3, a4, a5, v45, v46, v32, v27, v28);
  sub_2013400(a1, a2, 1, (__int64)v47, (__m128i *)(v37 & 0xFFFFFFFF00000000LL | v29), v30);
  if ( v48 )
    sub_161E7C0((__int64)&v48, v48);
  return v21;
}
