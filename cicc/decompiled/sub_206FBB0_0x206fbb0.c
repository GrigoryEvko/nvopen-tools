// Function: sub_206FBB0
// Address: 0x206fbb0
//
__int64 *__fastcall sub_206FBB0(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 **v6; // rax
  __int64 *v7; // rax
  __int64 *v8; // r12
  unsigned __int64 v9; // rdx
  __int64 (__fastcall *v10)(__int64, __int64); // r13
  __int64 v11; // rax
  unsigned int v12; // edx
  unsigned __int8 v13; // al
  int v14; // edx
  unsigned int v15; // r13d
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r12
  __int64 v22; // rdx
  __int64 v23; // r13
  __int64 v24; // rax
  __int64 v25; // rax
  const void **v26; // rdx
  __int64 *v27; // r11
  __int64 v28; // rcx
  __int64 v29; // rax
  const void **v30; // r8
  bool v31; // zf
  __int64 v32; // rsi
  __int64 *v33; // r12
  int v34; // edx
  int v35; // r13d
  __int64 *result; // rax
  __int128 v37; // [rsp-10h] [rbp-A0h]
  __int64 v38; // [rsp+8h] [rbp-88h]
  __int64 v39; // [rsp+8h] [rbp-88h]
  __int64 v40; // [rsp+10h] [rbp-80h]
  unsigned __int64 v41; // [rsp+18h] [rbp-78h]
  __int64 *v42; // [rsp+20h] [rbp-70h]
  const void **v43; // [rsp+20h] [rbp-70h]
  __int64 v44; // [rsp+28h] [rbp-68h]
  __int64 *v45; // [rsp+28h] [rbp-68h]
  __int64 v46; // [rsp+48h] [rbp-48h] BYREF
  __int64 v47; // [rsp+50h] [rbp-40h] BYREF
  int v48; // [rsp+58h] [rbp-38h]

  v44 = *(_QWORD *)(*(_QWORD *)(a1 + 552) + 16LL);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v6 = *(__int64 ***)(a2 - 8);
  else
    v6 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v7 = sub_20685E0(a1, *v6, a3, a4, a5);
  v8 = *(__int64 **)(a1 + 552);
  v40 = (__int64)v7;
  v41 = v9;
  v10 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v44 + 48LL);
  v11 = sub_1E0A0C0(v8[4]);
  if ( v10 == sub_1D13A20 )
  {
    v12 = 8 * sub_15A9520(v11, 0);
    if ( v12 == 32 )
    {
      v13 = 5;
    }
    else if ( v12 > 0x20 )
    {
      v13 = 6;
      if ( v12 != 64 )
      {
        v13 = 0;
        if ( v12 == 128 )
          v13 = 7;
      }
    }
    else
    {
      v13 = 3;
      if ( v12 != 8 )
        v13 = 4 * (v12 == 16);
    }
  }
  else
  {
    v13 = v10(v44, v11);
  }
  v14 = *(_DWORD *)(a1 + 536);
  v47 = 0;
  v15 = v13;
  v16 = *(_QWORD *)a1;
  v48 = v14;
  if ( v16 )
  {
    if ( &v47 != (__int64 *)(v16 + 48) )
    {
      v17 = *(_QWORD *)(v16 + 48);
      v47 = v17;
      if ( v17 )
        sub_1623A60((__int64)&v47, v17, 2);
    }
  }
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v18 = *(_QWORD *)(a2 - 8);
  else
    v18 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v19 = sub_20685E0(a1, *(__int64 **)(v18 + 24), a3, a4, a5);
  v21 = sub_1D322C0(
          v8,
          (__int64)v19,
          v20,
          (__int64)&v47,
          v15,
          0,
          *(double *)a3.m128i_i64,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128i_i64);
  v23 = v22;
  if ( v47 )
    sub_161E7C0((__int64)&v47, v47);
  v42 = *(__int64 **)(a1 + 552);
  v38 = *(_QWORD *)a2;
  v24 = sub_1E0A0C0(v42[4]);
  LOBYTE(v25) = sub_204D4D0(v44, v24, v38);
  v47 = 0;
  v27 = v42;
  v28 = v25;
  v29 = *(_QWORD *)a1;
  v30 = v26;
  v31 = *(_QWORD *)a1 == 0;
  v48 = *(_DWORD *)(a1 + 536);
  if ( !v31 && &v47 != (__int64 *)(v29 + 48) )
  {
    v32 = *(_QWORD *)(v29 + 48);
    v47 = v32;
    if ( v32 )
    {
      v39 = v28;
      v43 = v26;
      v45 = v27;
      sub_1623A60((__int64)&v47, v32, 2);
      v28 = v39;
      v30 = v43;
      v27 = v45;
    }
  }
  *((_QWORD *)&v37 + 1) = v23;
  *(_QWORD *)&v37 = v21;
  v33 = sub_1D332F0(
          v27,
          106,
          (__int64)&v47,
          v28,
          v30,
          0,
          *(double *)a3.m128i_i64,
          *(double *)a4.m128i_i64,
          a5,
          v40,
          v41,
          v37);
  v35 = v34;
  v46 = a2;
  result = sub_205F5C0(a1 + 8, &v46);
  result[1] = (__int64)v33;
  *((_DWORD *)result + 4) = v35;
  if ( v47 )
    return (__int64 *)sub_161E7C0((__int64)&v47, v47);
  return result;
}
