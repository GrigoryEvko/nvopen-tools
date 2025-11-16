// Function: sub_206D3A0
// Address: 0x206d3a0
//
__int64 *__fastcall sub_206D3A0(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  unsigned __int8 v6; // al
  unsigned int v7; // r15d
  __int64 **v8; // rax
  __int16 *v9; // rdx
  __int64 v10; // rax
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r13
  __int64 *v14; // r12
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rax
  unsigned int v18; // eax
  __int64 v19; // rcx
  __int64 *v20; // r10
  __int64 v21; // r9
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r8
  __int64 v25; // rdx
  bool v26; // zf
  __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 *v30; // r12
  int v31; // edx
  int v32; // r13d
  __int64 *result; // rax
  __int64 v34; // rsi
  __int128 v35; // [rsp-20h] [rbp-B0h]
  __int64 v36; // [rsp+0h] [rbp-90h]
  unsigned int v37; // [rsp+8h] [rbp-88h]
  const void **v38; // [rsp+8h] [rbp-88h]
  unsigned int v39; // [rsp+10h] [rbp-80h]
  __int64 *v40; // [rsp+10h] [rbp-80h]
  __int64 *v41; // [rsp+10h] [rbp-80h]
  __int64 v42; // [rsp+18h] [rbp-78h]
  unsigned int v43; // [rsp+18h] [rbp-78h]
  __int64 *v44; // [rsp+20h] [rbp-70h]
  __int16 *v45; // [rsp+28h] [rbp-68h]
  __int64 v46; // [rsp+48h] [rbp-48h] BYREF
  __int64 v47; // [rsp+50h] [rbp-40h] BYREF
  int v48; // [rsp+58h] [rbp-38h]

  v6 = *(_BYTE *)(a2 + 16);
  if ( v6 <= 0x17u )
  {
    if ( v6 == 5 )
      v7 = sub_1594720(a2);
    else
      v7 = 42;
  }
  else
  {
    v7 = 42;
    if ( v6 == 75 )
      v7 = *(_WORD *)(a2 + 18) & 0x7FFF;
  }
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v8 = *(__int64 ***)(a2 - 8);
  else
    v8 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v44 = sub_20685E0(a1, *v8, a3, a4, a5);
  v45 = v9;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v10 = *(_QWORD *)(a2 - 8);
  else
    v10 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v11 = sub_20685E0(a1, *(__int64 **)(v10 + 24), a3, a4, a5);
  v13 = v12;
  v14 = v11;
  v39 = sub_20C8390(v7);
  v15 = *(_QWORD *)(a1 + 552);
  v42 = *(_QWORD *)a2;
  v16 = *(_QWORD *)(v15 + 16);
  v17 = sub_1E0A0C0(*(_QWORD *)(v15 + 32));
  LOBYTE(v18) = sub_204D4D0(v16, v17, v42);
  v20 = *(__int64 **)(a1 + 552);
  v21 = v39;
  v47 = 0;
  v43 = v18;
  v22 = *(_QWORD *)a1;
  v24 = v23;
  v25 = *(unsigned int *)(a1 + 536);
  v26 = *(_QWORD *)a1 == 0;
  v48 = *(_DWORD *)(a1 + 536);
  if ( !v26 )
  {
    v25 = v22 + 48;
    if ( &v47 != (__int64 *)(v22 + 48) )
    {
      v27 = *(_QWORD *)(v22 + 48);
      v47 = v27;
      if ( v27 )
      {
        v40 = v20;
        v36 = v24;
        v37 = v21;
        sub_1623A60((__int64)&v47, v27, 2);
        v24 = v36;
        v21 = v37;
        v20 = v40;
      }
    }
  }
  v38 = (const void **)v24;
  v41 = v20;
  v28 = sub_1D28D50(v20, v21, v25, v19, v24, v21);
  *((_QWORD *)&v35 + 1) = v13;
  *(_QWORD *)&v35 = v14;
  v30 = sub_1D3A900(
          v41,
          0x89u,
          (__int64)&v47,
          v43,
          v38,
          0,
          (__m128)a3,
          *(double *)a4.m128i_i64,
          a5,
          (unsigned __int64)v44,
          v45,
          v35,
          v28,
          v29);
  v32 = v31;
  v46 = a2;
  result = sub_205F5C0(a1 + 8, &v46);
  v34 = v47;
  result[1] = (__int64)v30;
  *((_DWORD *)result + 4) = v32;
  if ( v34 )
    return (__int64 *)sub_161E7C0((__int64)&v47, v34);
  return result;
}
