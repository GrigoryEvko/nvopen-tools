// Function: sub_2918880
// Address: 0x2918880
//
__int64 __fastcall sub_2918880(__int64 *a1, __int64 a2, int a3, __int64 a4, const __m128i *a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // rbx
  int v8; // edx
  __int64 v10; // rax
  signed __int64 v11; // r15
  _BYTE *v12; // rcx
  int v13; // edx
  _BYTE *v14; // rdx
  char v15; // al
  unsigned int v16; // ebx
  __int64 v17; // rax
  __int64 v18; // rdi
  _BYTE *v19; // r10
  __int64 (__fastcall *v20)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v21; // rax
  _QWORD *v23; // rax
  __int64 v24; // rbx
  __int64 v25; // r12
  __int64 v26; // rdx
  unsigned int v27; // esi
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdi
  unsigned __int8 *v31; // rbx
  __int64 (__fastcall *v32)(__int64, _BYTE *, unsigned __int8 *); // rax
  _QWORD *v33; // rax
  __int64 v34; // rbx
  __int64 v35; // r12
  __int64 v36; // rdx
  unsigned int v37; // esi
  __int64 v38; // rax
  __int64 v39; // rcx
  __int64 v40; // [rsp+0h] [rbp-E0h]
  __int64 v41; // [rsp+8h] [rbp-D8h]
  _BYTE *v42; // [rsp+10h] [rbp-D0h]
  const __m128i *v43; // [rsp+10h] [rbp-D0h]
  __int64 v44; // [rsp+10h] [rbp-D0h]
  _BYTE *v45; // [rsp+10h] [rbp-D0h]
  _BYTE *v46; // [rsp+18h] [rbp-C8h]
  __m128i v47; // [rsp+20h] [rbp-C0h] BYREF
  char *v48; // [rsp+30h] [rbp-B0h]
  __int16 v49; // [rsp+40h] [rbp-A0h]
  __m128i v50[2]; // [rsp+50h] [rbp-90h] BYREF
  __int16 v51; // [rsp+70h] [rbp-70h]
  _BYTE *v52; // [rsp+80h] [rbp-60h] BYREF
  __int64 v53; // [rsp+88h] [rbp-58h]
  _BYTE v54[16]; // [rsp+90h] [rbp-50h] BYREF
  __int16 v55; // [rsp+A0h] [rbp-40h]

  v6 = a2;
  LODWORD(v7) = a3;
  v8 = a4 - a3;
  if ( v8 == *(_DWORD *)(*(_QWORD *)(a2 + 8) + 32LL) )
    return v6;
  if ( v8 != 1 )
  {
    v7 = (int)v7;
    v53 = 0x800000000LL;
    v10 = (int)a4;
    v52 = v54;
    v11 = (int)a4 - (__int64)(int)v7;
    if ( (unsigned __int64)v11 > 8 )
    {
      v41 = (int)a4;
      v43 = a5;
      sub_C8D5F0((__int64)&v52, v54, v11, 4u, (__int64)a5, a6);
      v13 = v53;
      a5 = v43;
      v46 = v52;
      v12 = &v52[4 * (unsigned int)v53];
      v10 = v41;
    }
    else
    {
      v46 = v54;
      v12 = v54;
      v13 = 0;
    }
    if ( v11 > 0 )
    {
      v14 = &v12[-4 * (int)v7];
      do
      {
        *(_DWORD *)&v14[4 * v7] = v7;
        ++v7;
      }
      while ( v10 != v7 );
      v13 = v53;
      v46 = v52;
    }
    v15 = a5[2].m128i_i8[0];
    v16 = v11 + v13;
    LODWORD(v53) = v11 + v13;
    if ( v15 )
    {
      if ( v15 == 1 )
      {
        v47.m128i_i64[0] = (__int64)".extract";
        v49 = 259;
      }
      else
      {
        if ( a5[2].m128i_i8[1] == 1 )
        {
          v39 = a5->m128i_i64[1];
          a5 = (const __m128i *)a5->m128i_i64[0];
          v40 = v39;
        }
        else
        {
          v15 = 2;
        }
        v47.m128i_i64[0] = (__int64)a5;
        LOBYTE(v49) = v15;
        v47.m128i_i64[1] = v40;
        v48 = ".extract";
        HIBYTE(v49) = 3;
      }
    }
    else
    {
      v49 = 256;
    }
    v17 = sub_ACADE0(*(__int64 ***)(a2 + 8));
    v18 = a1[10];
    v19 = (_BYTE *)v17;
    v20 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v18 + 112LL);
    if ( v20 == sub_9B6630 )
    {
      if ( *(_BYTE *)a2 > 0x15u || *v19 > 0x15u )
        goto LABEL_25;
      v42 = v19;
      v21 = sub_AD5CE0(a2, (__int64)v19, v46, v16, 0);
      v19 = v42;
      v6 = v21;
    }
    else
    {
      v45 = v19;
      v38 = ((__int64 (__fastcall *)(__int64, __int64, _BYTE *, _BYTE *, _QWORD))v20)(v18, a2, v19, v46, v16);
      v19 = v45;
      v6 = v38;
    }
    if ( v6 )
    {
LABEL_20:
      if ( v52 != v54 )
        _libc_free((unsigned __int64)v52);
      return v6;
    }
LABEL_25:
    v44 = (__int64)v19;
    v51 = 257;
    v23 = sub_BD2C40(112, unk_3F1FE60);
    v6 = (__int64)v23;
    if ( v23 )
      sub_B4E9E0((__int64)v23, a2, v44, v46, v16, (__int64)v50, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, __m128i *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v6,
      &v47,
      a1[7],
      a1[8]);
    v24 = *a1;
    v25 = *a1 + 16LL * *((unsigned int *)a1 + 2);
    while ( v25 != v24 )
    {
      v26 = *(_QWORD *)(v24 + 8);
      v27 = *(_DWORD *)v24;
      v24 += 16;
      sub_B99FD0(v6, v27, v26);
    }
    goto LABEL_20;
  }
  v47.m128i_i64[0] = (__int64)".extract";
  v49 = 259;
  sub_9C6370(v50, a5, &v47, a4, (__int64)a5, a6);
  v28 = sub_BCB2D0((_QWORD *)a1[9]);
  v29 = sub_ACD640(v28, (unsigned int)v7, 0);
  v30 = a1[10];
  v31 = (unsigned __int8 *)v29;
  v32 = *(__int64 (__fastcall **)(__int64, _BYTE *, unsigned __int8 *))(*(_QWORD *)v30 + 96LL);
  if ( v32 != sub_948070 )
  {
    v6 = v32(v30, (_BYTE *)a2, v31);
LABEL_34:
    if ( v6 )
      return v6;
    goto LABEL_35;
  }
  if ( *(_BYTE *)a2 <= 0x15u && *v31 <= 0x15u )
  {
    v6 = sub_AD5840(a2, v31, 0);
    goto LABEL_34;
  }
LABEL_35:
  v55 = 257;
  v33 = sub_BD2C40(72, 2u);
  v6 = (__int64)v33;
  if ( v33 )
    sub_B4DE80((__int64)v33, a2, (__int64)v31, (__int64)&v52, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, __m128i *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v6,
    v50,
    a1[7],
    a1[8]);
  v34 = *a1;
  v35 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  while ( v35 != v34 )
  {
    v36 = *(_QWORD *)(v34 + 8);
    v37 = *(_DWORD *)v34;
    v34 += 16;
    sub_B99FD0(v6, v37, v36);
  }
  return v6;
}
