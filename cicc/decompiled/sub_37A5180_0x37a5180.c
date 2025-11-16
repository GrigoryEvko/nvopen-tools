// Function: sub_37A5180
// Address: 0x37a5180
//
unsigned __int8 *__fastcall sub_37A5180(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v4; // rsi
  unsigned __int64 *v5; // rdx
  unsigned __int64 v6; // rsi
  __int64 v7; // rdx
  unsigned __int16 *v8; // rax
  __int64 v9; // r8
  unsigned int v10; // r10d
  unsigned __int16 *v11; // rax
  unsigned int v12; // r14d
  __int64 v13; // rdx
  __int128 v14; // rax
  _QWORD *v15; // r15
  __int128 v16; // rax
  __int64 v17; // r9
  unsigned int v18; // edx
  __int128 v19; // rax
  __int64 v20; // r9
  unsigned int v21; // edx
  unsigned int v22; // edx
  unsigned int v23; // edx
  unsigned __int8 *v24; // r14
  __int128 v26; // [rsp-10h] [rbp-D0h]
  __int64 v27; // [rsp+0h] [rbp-C0h]
  unsigned int v28; // [rsp+8h] [rbp-B8h]
  __int64 v29; // [rsp+10h] [rbp-B0h]
  unsigned int v30; // [rsp+18h] [rbp-A8h]
  _QWORD *v31; // [rsp+18h] [rbp-A8h]
  __int64 v32; // [rsp+18h] [rbp-A8h]
  __int128 v33; // [rsp+20h] [rbp-A0h]
  __int128 v34; // [rsp+30h] [rbp-90h]
  unsigned __int8 *v35; // [rsp+40h] [rbp-80h]
  __int64 v36; // [rsp+80h] [rbp-40h] BYREF
  int v37; // [rsp+88h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 80);
  v36 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v36, v4, 1);
  v5 = *(unsigned __int64 **)(a2 + 40);
  v37 = *(_DWORD *)(a2 + 72);
  v6 = *v5;
  v7 = v5[1];
  v8 = (unsigned __int16 *)(*(_QWORD *)(v6 + 48) + 16LL * (unsigned int)v7);
  v9 = *((_QWORD *)v8 + 1);
  v10 = *v8;
  v11 = *(unsigned __int16 **)(a2 + 48);
  v29 = v9;
  v12 = *v11;
  v30 = v10;
  v27 = *((_QWORD *)v11 + 1);
  *(_QWORD *)&v34 = sub_379AB60(a1, v6, v7);
  *((_QWORD *)&v34 + 1) = v13;
  *(_QWORD *)&v14 = sub_379AB60(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v15 = *(_QWORD **)(a1 + 8);
  v33 = v14;
  *(_QWORD *)&v16 = sub_3400EE0((__int64)v15, 0, (__int64)&v36, 0, a3);
  v28 = v30;
  sub_3406EB0(v15, 0xA1u, (__int64)&v36, v30, v29, v17, v34, v16);
  v31 = *(_QWORD **)(a1 + 8);
  *((_QWORD *)&v34 + 1) = v18 | *((_QWORD *)&v34 + 1) & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v19 = sub_3400EE0((__int64)v31, 0, (__int64)&v36, 0, a3);
  sub_3406EB0(v31, 0xA1u, (__int64)&v36, v28, v29, v20, v33, v19);
  v32 = v27;
  *((_QWORD *)&v33 + 1) = v21 | *((_QWORD *)&v33 + 1) & 0xFFFFFFFF00000000LL;
  LODWORD(v29) = (*(_DWORD *)(a2 + 24) != 184) + 213;
  *(_QWORD *)&v34 = sub_33FAF80(*(_QWORD *)(a1 + 8), (unsigned int)v29, (__int64)&v36, v12, v27, v27, a3);
  *((_QWORD *)&v34 + 1) = v22 | *((_QWORD *)&v34 + 1) & 0xFFFFFFFF00000000LL;
  v35 = sub_33FAF80(*(_QWORD *)(a1 + 8), (unsigned int)v29, (__int64)&v36, v12, v32, v32, a3);
  *((_QWORD *)&v26 + 1) = v23 | *((_QWORD *)&v33 + 1) & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v26 = v35;
  v24 = sub_3406EB0(*(_QWORD **)(a1 + 8), *(_DWORD *)(a2 + 24), (__int64)&v36, v12, v32, v32, v34, v26);
  if ( v36 )
    sub_B91220((__int64)&v36, v36);
  return v24;
}
