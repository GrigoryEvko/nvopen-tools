// Function: sub_203BA50
// Address: 0x203ba50
//
__int64 *__fastcall sub_203BA50(__int64 *a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  unsigned __int64 *v6; // rdx
  unsigned __int64 v7; // r14
  unsigned __int64 v8; // r15
  __int64 v9; // rax
  char v10; // cl
  __int64 v11; // rax
  __int64 *result; // rax
  bool v13; // al
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rsi
  __int64 *v18; // r10
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned int v21; // esi
  unsigned int v22; // eax
  __int64 v23; // rdx
  const void **v24; // rdx
  __int64 v25; // rax
  unsigned int v26; // edx
  __int64 *v27; // rax
  unsigned __int64 v28; // rdx
  unsigned int v29; // edx
  unsigned __int64 *v30; // [rsp+0h] [rbp-B0h]
  __int64 v31; // [rsp+0h] [rbp-B0h]
  const void **v32; // [rsp+0h] [rbp-B0h]
  __int64 v33; // [rsp+8h] [rbp-A8h]
  unsigned int v34; // [rsp+18h] [rbp-98h]
  __int64 *v35; // [rsp+18h] [rbp-98h]
  unsigned int v36; // [rsp+20h] [rbp-90h]
  __int128 v37; // [rsp+20h] [rbp-90h]
  unsigned int v38; // [rsp+20h] [rbp-90h]
  __int64 v39; // [rsp+30h] [rbp-80h]
  unsigned __int64 v40; // [rsp+38h] [rbp-78h]
  __int64 *v41; // [rsp+38h] [rbp-78h]
  unsigned int v42; // [rsp+40h] [rbp-70h] BYREF
  const void **v43; // [rsp+48h] [rbp-68h]
  _BYTE v44[8]; // [rsp+50h] [rbp-60h] BYREF
  __int64 v45; // [rsp+58h] [rbp-58h]
  __int64 v46; // [rsp+60h] [rbp-50h] BYREF
  int v47; // [rsp+68h] [rbp-48h]
  const void **v48; // [rsp+70h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v46,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  LOBYTE(v42) = v47;
  v43 = v48;
  if ( (_BYTE)v47 )
    v36 = word_4305480[(unsigned __int8)(v47 - 14)];
  else
    v36 = sub_1F58D30((__int64)&v42);
  v6 = *(unsigned __int64 **)(a2 + 32);
  v7 = *v6;
  v8 = v6[1];
  v34 = *((_DWORD *)v6 + 2);
  v39 = v34;
  v9 = *(_QWORD *)(*v6 + 40) + 16LL * v34;
  v40 = *v6;
  v10 = *(_BYTE *)v9;
  v11 = *(_QWORD *)(v9 + 8);
  v44[0] = v10;
  v45 = v11;
  if ( v10 )
  {
    if ( (unsigned __int8)(v10 - 14) > 0x5Fu )
      goto LABEL_8;
  }
  else
  {
    v30 = v6;
    v13 = sub_1F58D20((__int64)v44);
    v6 = v30;
    if ( !v13 )
      goto LABEL_8;
  }
  result = sub_203AD40(a1, a2);
  if ( result )
    return result;
  LOBYTE(v22) = sub_1F7E0F0((__int64)v44);
  v38 = sub_1F7DEB0(*(_QWORD **)(a1[1] + 48), v22, v23, v36, 0);
  v32 = v24;
  sub_1F40D10((__int64)&v46, *a1, *(_QWORD *)(a1[1] + 48), v44[0], v45);
  if ( (_BYTE)v46 == 7 )
  {
    v40 = sub_20363F0((__int64)a1, v7, v8);
    v34 = v29;
    v8 = v29 | v8 & 0xFFFFFFFF00000000LL;
  }
  sub_1F40D10((__int64)&v46, *a1, *(_QWORD *)(a1[1] + 48), v44[0], v45);
  if ( (_BYTE)v46 == 6 )
  {
    v27 = sub_2029F50((__int64)a1, a2);
    return sub_2030300(a1, (__int64)v27, v28, v42, v43, 0, a3, a4, a5);
  }
  v39 = v34;
  v25 = *(_QWORD *)(v40 + 40) + 16LL * v34;
  if ( *(_BYTE *)v25 != (_BYTE)v38 || *(const void ***)(v25 + 8) != v32 && !*(_BYTE *)v25 )
  {
    v8 = v34 | v8 & 0xFFFFFFFF00000000LL;
    v40 = (unsigned __int64)sub_2030300(a1, v40, v8, v38, v32, 0, a3, a4, a5);
    v39 = v26;
  }
  v6 = *(unsigned __int64 **)(a2 + 32);
LABEL_8:
  *(_QWORD *)&v37 = sub_20363F0((__int64)a1, v6[5], v6[6]);
  *((_QWORD *)&v37 + 1) = v14;
  v15 = sub_20363F0((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  v17 = *(_QWORD *)(a2 + 72);
  v18 = (__int64 *)a1[1];
  v19 = v15;
  v20 = v16;
  v46 = v17;
  if ( v17 )
  {
    v33 = v16;
    v35 = v18;
    v31 = v15;
    sub_1623A60((__int64)&v46, v17, 2);
    v19 = v31;
    v20 = v33;
    v18 = v35;
  }
  v21 = *(unsigned __int16 *)(a2 + 24);
  v47 = *(_DWORD *)(a2 + 64);
  result = sub_1D3A900(
             v18,
             v21,
             (__int64)&v46,
             v42,
             v43,
             0,
             (__m128)a3,
             a4,
             a5,
             v40,
             (__int16 *)(v8 & 0xFFFFFFFF00000000LL | v39),
             v37,
             v19,
             v20);
  if ( v46 )
  {
    v41 = result;
    sub_161E7C0((__int64)&v46, v46);
    return v41;
  }
  return result;
}
