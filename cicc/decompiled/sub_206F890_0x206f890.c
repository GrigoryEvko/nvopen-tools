// Function: sub_206F890
// Address: 0x206f890
//
__int64 *__fastcall sub_206F890(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 **v6; // rax
  __int16 *v7; // rdx
  __int64 v8; // rax
  __int128 v9; // rax
  __int64 *v10; // r12
  __int64 (__fastcall *v11)(__int64, __int64); // r13
  __int64 v12; // rax
  unsigned int v13; // edx
  unsigned __int8 v14; // al
  int v15; // edx
  unsigned int v16; // r13d
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r12
  __int64 v23; // rdx
  __int64 v24; // r13
  __int64 v25; // rax
  unsigned __int64 v26; // rax
  const void **v27; // rdx
  __int64 *v28; // r11
  unsigned __int64 v29; // rcx
  __int64 v30; // rax
  const void **v31; // r8
  bool v32; // zf
  __int64 v33; // rsi
  __int64 *v34; // r12
  int v35; // edx
  int v36; // r13d
  __int64 *result; // rax
  __int64 v38; // [rsp+8h] [rbp-98h]
  unsigned __int64 v39; // [rsp+8h] [rbp-98h]
  __int128 v40; // [rsp+10h] [rbp-90h]
  __int64 *v41; // [rsp+20h] [rbp-80h]
  __int16 *v42; // [rsp+28h] [rbp-78h]
  __int64 *v43; // [rsp+30h] [rbp-70h]
  const void **v44; // [rsp+30h] [rbp-70h]
  __int64 v45; // [rsp+38h] [rbp-68h]
  __int64 *v46; // [rsp+38h] [rbp-68h]
  __int64 v47; // [rsp+58h] [rbp-48h] BYREF
  __int64 v48; // [rsp+60h] [rbp-40h] BYREF
  int v49; // [rsp+68h] [rbp-38h]

  v45 = *(_QWORD *)(*(_QWORD *)(a1 + 552) + 16LL);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v6 = *(__int64 ***)(a2 - 8);
  else
    v6 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v41 = sub_20685E0(a1, *v6, a3, a4, a5);
  v42 = v7;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v8 = *(_QWORD *)(a2 - 8);
  else
    v8 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  *(_QWORD *)&v9 = sub_20685E0(a1, *(__int64 **)(v8 + 24), a3, a4, a5);
  v10 = *(__int64 **)(a1 + 552);
  v40 = v9;
  v11 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v45 + 48LL);
  v12 = sub_1E0A0C0(v10[4]);
  if ( v11 == sub_1D13A20 )
  {
    v13 = 8 * sub_15A9520(v12, 0);
    if ( v13 == 32 )
    {
      v14 = 5;
    }
    else if ( v13 > 0x20 )
    {
      v14 = 6;
      if ( v13 != 64 )
      {
        v14 = 0;
        if ( v13 == 128 )
          v14 = 7;
      }
    }
    else
    {
      v14 = 3;
      if ( v13 != 8 )
        v14 = 4 * (v13 == 16);
    }
  }
  else
  {
    v14 = v11(v45, v12);
  }
  v15 = *(_DWORD *)(a1 + 536);
  v48 = 0;
  v16 = v14;
  v17 = *(_QWORD *)a1;
  v49 = v15;
  if ( v17 )
  {
    if ( &v48 != (__int64 *)(v17 + 48) )
    {
      v18 = *(_QWORD *)(v17 + 48);
      v48 = v18;
      if ( v18 )
        sub_1623A60((__int64)&v48, v18, 2);
    }
  }
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v19 = *(_QWORD *)(a2 - 8);
  else
    v19 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v20 = sub_20685E0(a1, *(__int64 **)(v19 + 48), a3, a4, a5);
  v22 = sub_1D322C0(
          v10,
          (__int64)v20,
          v21,
          (__int64)&v48,
          v16,
          0,
          *(double *)a3.m128i_i64,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128i_i64);
  v24 = v23;
  if ( v48 )
    sub_161E7C0((__int64)&v48, v48);
  v43 = *(__int64 **)(a1 + 552);
  v38 = *(_QWORD *)a2;
  v25 = sub_1E0A0C0(v43[4]);
  LOBYTE(v26) = sub_204D4D0(v45, v25, v38);
  v48 = 0;
  v28 = v43;
  v29 = v26;
  v30 = *(_QWORD *)a1;
  v31 = v27;
  v32 = *(_QWORD *)a1 == 0;
  v49 = *(_DWORD *)(a1 + 536);
  if ( !v32 && &v48 != (__int64 *)(v30 + 48) )
  {
    v33 = *(_QWORD *)(v30 + 48);
    v48 = v33;
    if ( v33 )
    {
      v39 = v29;
      v44 = v27;
      v46 = v28;
      sub_1623A60((__int64)&v48, v33, 2);
      v29 = v39;
      v31 = v44;
      v28 = v46;
    }
  }
  v34 = sub_1D3A900(
          v28,
          0x69u,
          (__int64)&v48,
          v29,
          v31,
          0,
          (__m128)a3,
          *(double *)a4.m128i_i64,
          a5,
          (unsigned __int64)v41,
          v42,
          v40,
          v22,
          v24);
  v36 = v35;
  v47 = a2;
  result = sub_205F5C0(a1 + 8, &v47);
  result[1] = (__int64)v34;
  *((_DWORD *)result + 4) = v36;
  if ( v48 )
    return (__int64 *)sub_161E7C0((__int64)&v48, v48);
  return result;
}
