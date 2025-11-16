// Function: sub_206D5E0
// Address: 0x206d5e0
//
__int64 *__fastcall sub_206D5E0(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  unsigned __int8 v6; // al
  unsigned int v7; // r15d
  __int64 **v8; // rax
  __int16 *v9; // rdx
  __int64 v10; // rax
  __int64 *v11; // r12
  __int64 v12; // rdx
  __int64 v13; // r13
  unsigned int v14; // r9d
  unsigned __int8 v15; // al
  char v16; // dl
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 v19; // rax
  unsigned int v20; // eax
  __int64 v21; // rcx
  __int64 *v22; // r10
  __int64 v23; // r9
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r8
  __int64 v27; // rdx
  bool v28; // zf
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 *v32; // r12
  int v33; // edx
  int v34; // r13d
  __int64 *result; // rax
  __int64 v36; // rsi
  char v37; // al
  __int128 v38; // [rsp-20h] [rbp-B0h]
  __int64 v39; // [rsp+0h] [rbp-90h]
  unsigned int v40; // [rsp+8h] [rbp-88h]
  const void **v41; // [rsp+8h] [rbp-88h]
  unsigned int v42; // [rsp+10h] [rbp-80h]
  __int64 *v43; // [rsp+10h] [rbp-80h]
  __int64 *v44; // [rsp+10h] [rbp-80h]
  __int64 v45; // [rsp+18h] [rbp-78h]
  unsigned int v46; // [rsp+18h] [rbp-78h]
  __int64 *v47; // [rsp+20h] [rbp-70h]
  __int16 *v48; // [rsp+28h] [rbp-68h]
  __int64 v49; // [rsp+48h] [rbp-48h] BYREF
  __int64 v50; // [rsp+50h] [rbp-40h] BYREF
  int v51; // [rsp+58h] [rbp-38h]

  v6 = *(_BYTE *)(a2 + 16);
  if ( v6 <= 0x17u )
  {
    if ( v6 == 5 )
      v7 = sub_1594720(a2);
    else
      v7 = 16;
  }
  else
  {
    v7 = 16;
    if ( v6 == 76 )
      v7 = *(_WORD *)(a2 + 18) & 0x7FFF;
  }
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v8 = *(__int64 ***)(a2 - 8);
  else
    v8 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v47 = sub_20685E0(a1, *v8, a3, a4, a5);
  v48 = v9;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v10 = *(_QWORD *)(a2 - 8);
  else
    v10 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v11 = sub_20685E0(a1, *(__int64 **)(v10 + 24), a3, a4, a5);
  v13 = v12;
  v14 = sub_20C82F0(v7);
  v15 = *(_BYTE *)(a2 + 16);
  if ( v15 <= 0x17u )
  {
    if ( v15 != 5 )
      goto LABEL_22;
    v37 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
    if ( v37 == 16 )
      v37 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a2 + 16LL) + 8LL);
    if ( (unsigned __int8)(v37 - 1) > 5u && *(_WORD *)(a2 + 18) != 52 )
    {
LABEL_22:
      if ( (*(_BYTE *)(*(_QWORD *)(a1 + 544) + 792LL) & 8) == 0 )
        goto LABEL_14;
      goto LABEL_13;
    }
  }
  else
  {
    v16 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
    if ( v16 == 16 )
    {
      if ( (unsigned __int8)(*(_BYTE *)(**(_QWORD **)(*(_QWORD *)a2 + 16LL) + 8LL) - 1) <= 5u )
        goto LABEL_12;
    }
    else if ( (unsigned __int8)(v16 - 1) <= 5u )
    {
      goto LABEL_12;
    }
    if ( v15 != 76 )
      goto LABEL_22;
  }
LABEL_12:
  if ( (*(_BYTE *)(a2 + 17) & 4) == 0 )
    goto LABEL_22;
LABEL_13:
  v14 = sub_20C8300(v14);
LABEL_14:
  v17 = *(_QWORD *)(a1 + 552);
  v42 = v14;
  v18 = *(_QWORD *)(v17 + 16);
  v45 = *(_QWORD *)a2;
  v19 = sub_1E0A0C0(*(_QWORD *)(v17 + 32));
  LOBYTE(v20) = sub_204D4D0(v18, v19, v45);
  v22 = *(__int64 **)(a1 + 552);
  v23 = v42;
  v50 = 0;
  v46 = v20;
  v24 = *(_QWORD *)a1;
  v26 = v25;
  v27 = *(unsigned int *)(a1 + 536);
  v28 = *(_QWORD *)a1 == 0;
  v51 = *(_DWORD *)(a1 + 536);
  if ( !v28 )
  {
    v27 = v24 + 48;
    if ( &v50 != (__int64 *)(v24 + 48) )
    {
      v29 = *(_QWORD *)(v24 + 48);
      v50 = v29;
      if ( v29 )
      {
        v43 = v22;
        v39 = v26;
        v40 = v23;
        sub_1623A60((__int64)&v50, v29, 2);
        v26 = v39;
        v23 = v40;
        v22 = v43;
      }
    }
  }
  v41 = (const void **)v26;
  v44 = v22;
  v30 = sub_1D28D50(v22, v23, v27, v21, v26, v23);
  *((_QWORD *)&v38 + 1) = v13;
  *(_QWORD *)&v38 = v11;
  v32 = sub_1D3A900(
          v44,
          0x89u,
          (__int64)&v50,
          v46,
          v41,
          0,
          (__m128)a3,
          *(double *)a4.m128i_i64,
          a5,
          (unsigned __int64)v47,
          v48,
          v38,
          v30,
          v31);
  v34 = v33;
  v49 = a2;
  result = sub_205F5C0(a1 + 8, &v49);
  v36 = v50;
  result[1] = (__int64)v32;
  *((_DWORD *)result + 4) = v34;
  if ( v36 )
    return (__int64 *)sub_161E7C0((__int64)&v50, v36);
  return result;
}
