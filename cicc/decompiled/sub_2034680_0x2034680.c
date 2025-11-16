// Function: sub_2034680
// Address: 0x2034680
//
__int64 *__fastcall sub_2034680(__int64 a1, __int64 a2, __m128 a3, double a4, __m128i a5)
{
  __int64 v6; // r14
  __int64 v7; // rdx
  unsigned int v8; // r15d
  __int64 v9; // rax
  __int64 v10; // rsi
  unsigned __int64 *v11; // rcx
  __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // r13
  __int64 v15; // rax
  unsigned __int8 v16; // r8
  const void **v17; // rax
  __int64 v18; // r9
  unsigned int v19; // edx
  __int64 v20; // r8
  unsigned __int64 v21; // r10
  __int16 *v22; // r11
  __int64 v23; // rax
  char v24; // cl
  __int64 v25; // rax
  unsigned int v26; // esi
  __int64 *v27; // r12
  bool v29; // al
  __int128 v30; // [rsp-20h] [rbp-B0h]
  unsigned __int64 *v31; // [rsp+8h] [rbp-88h]
  unsigned int v32; // [rsp+8h] [rbp-88h]
  unsigned __int8 v33; // [rsp+10h] [rbp-80h]
  unsigned __int64 v34; // [rsp+10h] [rbp-80h]
  __int16 *v35; // [rsp+18h] [rbp-78h]
  __int64 v36; // [rsp+20h] [rbp-70h]
  const void **v37; // [rsp+30h] [rbp-60h]
  __int64 *v38; // [rsp+38h] [rbp-58h]
  __int64 v39; // [rsp+40h] [rbp-50h] BYREF
  int v40; // [rsp+48h] [rbp-48h]
  _BYTE v41[8]; // [rsp+50h] [rbp-40h] BYREF
  __int64 v42; // [rsp+58h] [rbp-38h]

  v6 = sub_2032580(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
  v8 = v7;
  v36 = v7;
  v38 = *(__int64 **)(a1 + 8);
  v9 = sub_2032580(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  v10 = *(_QWORD *)(a2 + 72);
  v11 = *(unsigned __int64 **)(a2 + 32);
  v12 = v9;
  v14 = v13;
  v15 = *(_QWORD *)(v6 + 40) + 16LL * v8;
  v16 = *(_BYTE *)v15;
  v17 = *(const void ***)(v15 + 8);
  v39 = v10;
  v37 = v17;
  if ( v10 )
  {
    v31 = v11;
    v33 = v16;
    sub_1623A60((__int64)&v39, v10, 2);
    v11 = v31;
    v16 = v33;
  }
  v18 = v36;
  v19 = v16;
  v20 = v6;
  v40 = *(_DWORD *)(a2 + 64);
  v21 = *v11;
  v22 = (__int16 *)v11[1];
  v23 = *(_QWORD *)(*v11 + 40) + 16LL * *((unsigned int *)v11 + 2);
  v24 = *(_BYTE *)v23;
  v25 = *(_QWORD *)(v23 + 8);
  v41[0] = v24;
  v42 = v25;
  if ( v24 )
  {
    v26 = ((unsigned __int8)(v24 - 14) < 0x60u) + 134;
  }
  else
  {
    v32 = v19;
    v34 = v21;
    v35 = v22;
    v29 = sub_1F58D20((__int64)v41);
    v19 = v32;
    v21 = v34;
    v22 = v35;
    v20 = v6;
    v18 = v36;
    v26 = 134 - (!v29 - 1);
  }
  *((_QWORD *)&v30 + 1) = v18;
  *(_QWORD *)&v30 = v20;
  v27 = sub_1D3A900(v38, v26, (__int64)&v39, v19, v37, 0, a3, a4, a5, v21, v22, v30, v12, v14);
  if ( v39 )
    sub_161E7C0((__int64)&v39, v39);
  return v27;
}
