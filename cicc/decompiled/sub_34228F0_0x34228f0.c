// Function: sub_34228F0
// Address: 0x34228f0
//
void __fastcall sub_34228F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rdx
  unsigned __int8 v5; // al
  __int64 *v6; // rax
  __int64 v7; // r9
  __int64 v8; // r14
  __int64 v9; // r13
  __int64 v10; // rax
  int v11; // eax
  __int64 *v12; // rdx
  __int64 *v13; // r13
  unsigned __int16 *v14; // rax
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // r9
  unsigned __int8 *v20; // rax
  __int64 v21; // r13
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rax
  __int128 v26; // [rsp-10h] [rbp-A0h]
  int v27; // [rsp+4h] [rbp-8Ch]
  __int64 v28; // [rsp+8h] [rbp-88h]
  unsigned int *v29; // [rsp+10h] [rbp-80h]
  __int64 v30; // [rsp+18h] [rbp-78h]
  __int64 (__fastcall *v31)(__int64, __int64, __int64, __int64); // [rsp+20h] [rbp-70h]
  __int64 v32; // [rsp+28h] [rbp-68h]
  __int64 v33; // [rsp+28h] [rbp-68h]
  __int64 v34; // [rsp+28h] [rbp-68h]
  __int64 v35; // [rsp+30h] [rbp-60h] BYREF
  int v36; // [rsp+38h] [rbp-58h]
  __int64 v37; // [rsp+40h] [rbp-50h] BYREF
  int v38; // [rsp+48h] [rbp-48h]
  _QWORD *v39; // [rsp+50h] [rbp-40h]
  __int64 v40; // [rsp+58h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 80);
  v35 = v3;
  if ( v3 )
    sub_B96E90((__int64)&v35, v3, 1);
  v36 = *(_DWORD *)(a2 + 72);
  v4 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 96LL);
  v5 = *(_BYTE *)(v4 - 16);
  if ( (v5 & 2) != 0 )
    v6 = *(__int64 **)(v4 - 32);
  else
    v6 = (__int64 *)(v4 - 8LL * ((v5 >> 2) & 0xF) - 16);
  v7 = *v6;
  v8 = 0;
  if ( **(_WORD **)(a2 + 48) )
  {
    v34 = *v6;
    v25 = sub_350FCC0();
    v7 = v34;
    v8 = v25;
  }
  v9 = *(_QWORD *)(a1 + 808);
  v31 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v9 + 2352LL);
  v32 = *(_QWORD *)(*(_QWORD *)(a1 + 64) + 40LL);
  v10 = sub_B91420(v7);
  v11 = v31(v9, v10, v8, v32);
  v12 = *(__int64 **)(a2 + 40);
  v13 = *(__int64 **)(a1 + 64);
  v27 = v11;
  v14 = *(unsigned __int16 **)(a2 + 48);
  v15 = *v14;
  v33 = *v12;
  LODWORD(v31) = *((_DWORD *)v12 + 2);
  v28 = *((_QWORD *)v14 + 1);
  v16 = sub_33E5110(v13, v15, v28, 1, 0);
  v30 = v17;
  v37 = v33;
  v29 = (unsigned int *)v16;
  v38 = (int)v31;
  v39 = sub_33F0B60(v13, v27, v15, v28);
  v40 = v18;
  *((_QWORD *)&v26 + 1) = 2;
  *(_QWORD *)&v26 = &v37;
  v20 = sub_3411630(v13, 50, (__int64)&v35, v29, v30, v19, v26);
  *((_DWORD *)v20 + 9) = -1;
  v21 = (__int64)v20;
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, (__int64)v20, v22, v23, v24);
  sub_3421DB0(v21);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v35 )
    sub_B91220((__int64)&v35, v35);
}
