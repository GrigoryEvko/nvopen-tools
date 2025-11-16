// Function: sub_3422AE0
// Address: 0x3422ae0
//
void __fastcall sub_3422AE0(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rdx
  __int64 v5; // rcx
  unsigned __int8 v6; // al
  __int64 *v7; // rax
  __int64 v8; // r9
  __int64 v9; // r15
  __int64 v10; // r14
  __int64 v11; // rax
  int v12; // esi
  __int64 *v13; // rax
  __int64 v14; // r14
  __int64 v15; // r15
  __int128 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rax
  __int128 v23; // [rsp-30h] [rbp-90h]
  __int64 (__fastcall *v24)(__int64, __int64, __int64, __int64); // [rsp+0h] [rbp-60h]
  __int128 v25; // [rsp+0h] [rbp-60h]
  __int64 v26; // [rsp+18h] [rbp-48h]
  _QWORD *v27; // [rsp+18h] [rbp-48h]
  __int64 v28; // [rsp+18h] [rbp-48h]
  __int64 v29; // [rsp+20h] [rbp-40h] BYREF
  int v30; // [rsp+28h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 80);
  v29 = v3;
  if ( v3 )
    sub_B96E90((__int64)&v29, v3, 1);
  v4 = *(_QWORD *)(a2 + 40);
  v30 = *(_DWORD *)(a2 + 72);
  v5 = *(_QWORD *)(*(_QWORD *)(v4 + 40) + 96LL);
  v6 = *(_BYTE *)(v5 - 16);
  if ( (v6 & 2) != 0 )
    v7 = *(__int64 **)(v5 - 32);
  else
    v7 = (__int64 *)(v5 - 8LL * ((v6 >> 2) & 0xF) - 16);
  v8 = *v7;
  v9 = 0;
  if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)(v4 + 80) + 48LL) + 16LL * *(unsigned int *)(v4 + 88)) )
  {
    v28 = *v7;
    v22 = sub_350FCC0();
    v8 = v28;
    v9 = v22;
  }
  v10 = *(_QWORD *)(a1 + 808);
  v24 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v10 + 2352LL);
  v26 = *(_QWORD *)(*(_QWORD *)(a1 + 64) + 40LL);
  v11 = sub_B91420(v8);
  v12 = v24(v10, v11, v9, v26);
  v13 = *(__int64 **)(a2 + 40);
  v27 = *(_QWORD **)(a1 + 64);
  v14 = *v13;
  v15 = v13[1];
  v25 = *((_OWORD *)v13 + 5);
  *(_QWORD *)&v16 = sub_33F0B60(
                      v27,
                      v12,
                      *(unsigned __int16 *)(*(_QWORD *)(v25 + 48) + 16LL * *((unsigned int *)v13 + 22)),
                      *(_QWORD *)(*(_QWORD *)(v25 + 48) + 16LL * *((unsigned int *)v13 + 22) + 8));
  *((_QWORD *)&v23 + 1) = v15;
  *(_QWORD *)&v23 = v14;
  v17 = sub_340F900(v27, 0x31u, (__int64)&v29, 1u, 0, *((__int64 *)&v25 + 1), v23, v16, v25);
  *(_DWORD *)(v17 + 36) = -1;
  v18 = v17;
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v17, v19, v20, v21);
  sub_3421DB0(v18);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v29 )
    sub_B91220((__int64)&v29, v29);
}
