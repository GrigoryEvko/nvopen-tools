// Function: sub_206F370
// Address: 0x206f370
//
__int64 *__fastcall sub_206F370(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 **v6; // rax
  __int64 *v7; // rax
  __int64 *v8; // r12
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rax
  const void **v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // r12
  int v22; // edx
  int v23; // r13d
  __int64 *result; // rax
  __int64 v25; // r15
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rbx
  int v29; // edx
  int v30; // r12d
  __int64 *v31; // rax
  int v32; // edx
  __int64 v33; // rax
  _QWORD *v34; // r9
  unsigned int v35; // r8d
  __int64 v36; // rsi
  __int64 v37; // r12
  __int64 *v38; // rax
  __int128 v39; // [rsp-10h] [rbp-A0h]
  __int64 *v40; // [rsp-10h] [rbp-A0h]
  __int128 v41; // [rsp-10h] [rbp-A0h]
  unsigned int v42; // [rsp+0h] [rbp-90h]
  __int64 v43; // [rsp+8h] [rbp-88h]
  _QWORD *v44; // [rsp+8h] [rbp-88h]
  __int64 v45; // [rsp+40h] [rbp-50h] BYREF
  int v46; // [rsp+48h] [rbp-48h]
  __int64 v47; // [rsp+50h] [rbp-40h] BYREF
  int v48; // [rsp+58h] [rbp-38h]

  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v6 = *(__int64 ***)(a2 - 8);
  else
    v6 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v7 = sub_20685E0(a1, *v6, a3, a4, a5);
  v45 = 0;
  v8 = v7;
  v10 = v9;
  v11 = *(_QWORD *)a1;
  v46 = *(_DWORD *)(a1 + 536);
  if ( v11 )
  {
    if ( &v45 != (__int64 *)(v11 + 48) )
    {
      v12 = *(_QWORD *)(v11 + 48);
      v45 = v12;
      if ( v12 )
        sub_1623A60((__int64)&v45, v12, 2);
    }
  }
  v13 = *(_QWORD *)(a1 + 552);
  v14 = *(_QWORD *)(v13 + 16);
  v43 = *(_QWORD *)a2;
  v15 = sub_1E0A0C0(*(_QWORD *)(v13 + 32));
  LOBYTE(v16) = sub_204D4D0(v14, v15, v43);
  v18 = v16;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 15 || *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a2 + 16LL) + 8LL) != 12 )
  {
    v19 = v8[5] + 16LL * (unsigned int)v10;
    if ( *(_BYTE *)v19 != (_BYTE)v18 || !(_BYTE)v18 && *(const void ***)(v19 + 8) != v17 )
    {
      *((_QWORD *)&v39 + 1) = v10;
      *(_QWORD *)&v39 = v8;
      v20 = sub_1D309E0(
              *(__int64 **)(a1 + 552),
              158,
              (__int64)&v45,
              v18,
              v17,
              0,
              *(double *)a3.m128i_i64,
              *(double *)a4.m128i_i64,
              *(double *)a5.m128i_i64,
              v39);
      v47 = a2;
      v21 = v20;
      v23 = v22;
      result = sub_205F5C0(a1 + 8, &v47);
      result[1] = v21;
      *((_DWORD *)result + 4) = v23;
      goto LABEL_10;
    }
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    {
      v38 = *(__int64 **)(a2 - 8);
      v25 = a1 + 8;
      v26 = *v38;
      if ( *(_BYTE *)(*v38 + 16) == 13 )
        goto LABEL_16;
    }
    else
    {
      v25 = a1 + 8;
      v26 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      if ( *(_BYTE *)(v26 + 16) == 13 )
      {
LABEL_16:
        v27 = sub_1D38970(
                *(_QWORD *)(a1 + 552),
                v26 + 24,
                (__int64)&v45,
                v18,
                v17,
                0,
                a3,
                *(double *)a4.m128i_i64,
                a5,
                1u);
        v47 = a2;
        v28 = v27;
        v30 = v29;
        v31 = sub_205F5C0(v25, &v47);
        v31[1] = v28;
        *((_DWORD *)v31 + 4) = v30;
        result = v40;
        goto LABEL_10;
      }
    }
    v47 = a2;
    result = sub_205F5C0(v25, &v47);
    result[1] = (__int64)v8;
    *((_DWORD *)result + 4) = v10;
    goto LABEL_10;
  }
  v32 = *(_DWORD *)(a1 + 536);
  v33 = *(_QWORD *)a1;
  v47 = 0;
  v34 = *(_QWORD **)(a1 + 552);
  v35 = (unsigned __int8)v18;
  v48 = v32;
  if ( v33 )
  {
    if ( &v47 != (__int64 *)(v33 + 48) )
    {
      v36 = *(_QWORD *)(v33 + 48);
      v47 = v36;
      if ( v36 )
      {
        v42 = (unsigned __int8)v18;
        v44 = v34;
        sub_1623A60((__int64)&v47, v36, 2);
        v35 = v42;
        v34 = v44;
      }
    }
  }
  *((_QWORD *)&v41 + 1) = v10;
  *(_QWORD *)&v41 = v8;
  v37 = sub_1D2CC80(v34, 15, (__int64)&v47, v35, 0, (__int64)v34, v41);
  if ( v47 )
    sub_161E7C0((__int64)&v47, v47);
  v47 = a2;
  result = sub_205F5C0(a1 + 8, &v47);
  result[1] = v37;
  *((_DWORD *)result + 4) = 0;
LABEL_10:
  if ( v45 )
    return (__int64 *)sub_161E7C0((__int64)&v45, v45);
  return result;
}
