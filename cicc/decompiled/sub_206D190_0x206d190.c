// Function: sub_206D190
// Address: 0x206d190
//
__int64 *__fastcall sub_206D190(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 **v6; // rax
  __int64 *v7; // r14
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // r15
  __int64 v10; // rax
  __int64 *v11; // rax
  unsigned __int64 v12; // r9
  __int64 v13; // rdx
  __int64 v14; // r11
  int v15; // edx
  __int64 *v16; // r10
  bool v17; // al
  int v18; // edx
  __int64 v19; // rcx
  const void **v20; // r8
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 *v23; // r14
  int v24; // edx
  int v25; // r15d
  __int64 *result; // rax
  __int64 v27; // rsi
  __int128 v28; // [rsp-10h] [rbp-A0h]
  __int64 v29; // [rsp+0h] [rbp-90h]
  unsigned int v30; // [rsp+Ch] [rbp-84h]
  __int64 *v31; // [rsp+10h] [rbp-80h]
  __int64 v32; // [rsp+18h] [rbp-78h]
  const void **v33; // [rsp+20h] [rbp-70h]
  __int64 *v34; // [rsp+28h] [rbp-68h]
  __int64 v35; // [rsp+48h] [rbp-48h] BYREF
  __int64 v36; // [rsp+50h] [rbp-40h] BYREF
  int v37; // [rsp+58h] [rbp-38h]

  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v6 = *(__int64 ***)(a2 - 8);
  else
    v6 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v7 = sub_20685E0(a1, *v6, a3, a4, a5);
  v9 = v8;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v10 = *(_QWORD *)(a2 - 8);
  else
    v10 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v11 = sub_20685E0(a1, *(__int64 **)(v10 + 24), a3, a4, a5);
  v12 = 0;
  v14 = v13;
  v15 = *(unsigned __int8 *)(a2 + 16);
  v16 = v11;
  if ( (unsigned __int8)v15 <= 0x17u )
  {
    v17 = 0;
    if ( (_BYTE)v15 == 5 )
    {
      v17 = (unsigned __int16)(*(_WORD *)(a2 + 18) - 24) <= 1u || (unsigned int)*(unsigned __int16 *)(a2 + 18) - 17 <= 1;
      if ( v17 )
        v17 = (*(_BYTE *)(a2 + 17) & 2) != 0;
    }
  }
  else
  {
    v17 = (unsigned int)(v15 - 41) <= 1 || (unsigned __int8)(v15 - 48) <= 1u;
    if ( v17 )
      v17 = (*(_BYTE *)(a2 + 17) & 2) != 0;
  }
  v18 = *(_DWORD *)(a1 + 536);
  LOBYTE(v12) = (8 * v17 + 1) & 9;
  v34 = *(__int64 **)(a1 + 552);
  v19 = *(unsigned __int8 *)(v7[5] + 16LL * (unsigned int)v9);
  v20 = *(const void ***)(v7[5] + 16LL * (unsigned int)v9 + 8);
  v36 = 0;
  v21 = *(_QWORD *)a1;
  v37 = v18;
  if ( v21 )
  {
    if ( &v36 != (__int64 *)(v21 + 48) )
    {
      v22 = *(_QWORD *)(v21 + 48);
      v36 = v22;
      if ( v22 )
      {
        v31 = v16;
        v29 = v19;
        v30 = v12;
        v32 = v14;
        v33 = v20;
        sub_1623A60((__int64)&v36, v22, 2);
        v19 = v29;
        v12 = v30;
        v16 = v31;
        v14 = v32;
        v20 = v33;
      }
    }
  }
  *((_QWORD *)&v28 + 1) = v14;
  *(_QWORD *)&v28 = v16;
  v23 = sub_1D332F0(
          v34,
          55,
          (__int64)&v36,
          v19,
          v20,
          v12,
          *(double *)a3.m128i_i64,
          *(double *)a4.m128i_i64,
          a5,
          (__int64)v7,
          v9,
          v28);
  v25 = v24;
  v35 = a2;
  result = sub_205F5C0(a1 + 8, &v35);
  v27 = v36;
  result[1] = (__int64)v23;
  *((_DWORD *)result + 4) = v25;
  if ( v27 )
    return (__int64 *)sub_161E7C0((__int64)&v36, v27);
  return result;
}
