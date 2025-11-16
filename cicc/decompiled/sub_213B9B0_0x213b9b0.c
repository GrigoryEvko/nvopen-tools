// Function: sub_213B9B0
// Address: 0x213b9b0
//
__int64 *__fastcall sub_213B9B0(__int64 a1, __int64 a2, __m128 a3, double a4, __m128i a5)
{
  __int64 v6; // r14
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rsi
  unsigned __int64 *v10; // rcx
  __int64 v11; // r12
  __int64 v12; // rdx
  __int64 v13; // r13
  __int64 v14; // rax
  unsigned __int8 v15; // r8
  const void **v16; // rax
  __int64 v17; // r9
  unsigned int v18; // edx
  __int64 v19; // r8
  unsigned __int64 v20; // r10
  __int16 *v21; // r11
  __int64 v22; // rax
  char v23; // cl
  __int64 v24; // rax
  unsigned int v25; // esi
  __int64 *v26; // r12
  bool v28; // al
  __int128 v29; // [rsp-20h] [rbp-B0h]
  unsigned __int64 *v30; // [rsp+8h] [rbp-88h]
  unsigned int v31; // [rsp+8h] [rbp-88h]
  unsigned __int8 v32; // [rsp+10h] [rbp-80h]
  unsigned __int64 v33; // [rsp+10h] [rbp-80h]
  __int16 *v34; // [rsp+18h] [rbp-78h]
  const void **v35; // [rsp+20h] [rbp-70h]
  __int64 *v36; // [rsp+28h] [rbp-68h]
  __int64 v37; // [rsp+30h] [rbp-60h]
  __int64 v38; // [rsp+40h] [rbp-50h] BYREF
  int v39; // [rsp+48h] [rbp-48h]
  _BYTE v40[8]; // [rsp+50h] [rbp-40h] BYREF
  __int64 v41; // [rsp+58h] [rbp-38h]

  v6 = sub_2138AD0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
  v37 = v7;
  v8 = sub_2138AD0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  v9 = *(_QWORD *)(a2 + 72);
  v10 = *(unsigned __int64 **)(a2 + 32);
  v11 = v8;
  v13 = v12;
  v36 = *(__int64 **)(a1 + 8);
  v14 = *(_QWORD *)(v6 + 40) + 16LL * (unsigned int)v37;
  v15 = *(_BYTE *)v14;
  v16 = *(const void ***)(v14 + 8);
  v38 = v9;
  v35 = v16;
  if ( v9 )
  {
    v30 = v10;
    v32 = v15;
    sub_1623A60((__int64)&v38, v9, 2);
    v10 = v30;
    v15 = v32;
  }
  v17 = v37;
  v18 = v15;
  v19 = v6;
  v39 = *(_DWORD *)(a2 + 64);
  v20 = *v10;
  v21 = (__int16 *)v10[1];
  v22 = *(_QWORD *)(*v10 + 40) + 16LL * *((unsigned int *)v10 + 2);
  v23 = *(_BYTE *)v22;
  v24 = *(_QWORD *)(v22 + 8);
  v40[0] = v23;
  v41 = v24;
  if ( v23 )
  {
    v25 = ((unsigned __int8)(v23 - 14) < 0x60u) + 134;
  }
  else
  {
    v31 = v18;
    v33 = v20;
    v34 = v21;
    v28 = sub_1F58D20((__int64)v40);
    v18 = v31;
    v20 = v33;
    v21 = v34;
    v19 = v6;
    v17 = v37;
    v25 = 134 - (!v28 - 1);
  }
  *((_QWORD *)&v29 + 1) = v17;
  *(_QWORD *)&v29 = v19;
  v26 = sub_1D3A900(v36, v25, (__int64)&v38, v18, v35, 0, a3, a4, a5, v20, v21, v29, v11, v13);
  if ( v38 )
    sub_161E7C0((__int64)&v38, v38);
  return v26;
}
