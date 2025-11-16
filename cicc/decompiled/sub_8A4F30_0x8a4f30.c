// Function: sub_8A4F30
// Address: 0x8a4f30
//
__int64 __fastcall sub_8A4F30(
        __int64 a1,
        __int64 a2,
        __m128i *a3,
        __int64 a4,
        __m128i *a5,
        __int64 a6,
        __int64 *a7,
        unsigned int a8,
        int *a9,
        __m128i *a10)
{
  __m128i *v10; // r10
  __int64 v11; // r11
  __int64 **v13; // r13
  __int32 v15; // r15d
  char v16; // dl
  int v17; // eax
  __int64 result; // rax
  char v19; // al
  __int64 **v20; // rax
  __int64 **v21; // rax
  __int64 **v22; // rdi
  unsigned int v23; // r9d
  __int64 v24; // rax
  __int64 v25; // r8
  __int64 v26; // rsi
  __int64 **v27; // rdi
  __int64 **v28; // rax
  __int64 v29; // rcx
  __int64 **v30; // rax
  __int64 v31; // rsi
  char v32; // dl
  bool v33; // zf
  const __m128i *v34; // r14
  __int64 **v35; // r13
  __m128i *v36; // rax
  int v37; // edi
  __int64 v38; // [rsp-8h] [rbp-98h]
  __int64 v42; // [rsp+8h] [rbp-88h]
  __int64 **v44; // [rsp+10h] [rbp-80h]
  __int64 v45; // [rsp+10h] [rbp-80h]
  __int64 v47; // [rsp+18h] [rbp-78h]
  int v48; // [rsp+28h] [rbp-68h] BYREF
  int v49; // [rsp+2Ch] [rbp-64h] BYREF
  __int64 **v50; // [rsp+30h] [rbp-60h] BYREF
  __m128i *v51; // [rsp+38h] [rbp-58h] BYREF
  _DWORD v52[20]; // [rsp+40h] [rbp-50h] BYREF

  v10 = a5;
  v11 = a6;
  v13 = (__int64 **)a4;
  v15 = a10[5].m128i_i32[2];
  if ( a4 )
  {
    v16 = *(_BYTE *)(*(_QWORD *)(a2 + 8) + 80LL);
    v17 = 0;
    if ( v16 != 3 )
      v17 = (v16 != 2) + 1;
    if ( *(unsigned __int8 *)(a1 + 8) != v17 )
    {
      result = (__int64)a9;
      *a9 = 1;
      v15 |= a10[5].m128i_u32[2];
      goto LABEL_6;
    }
  }
  a10[5].m128i_i32[2] = 0;
  v19 = *(_BYTE *)(a1 + 8);
  if ( !v19 )
  {
    v21 = sub_8A2270(*(_QWORD *)(a1 + 32), a5, a6, a7, a8, a9, a10);
    *(_QWORD *)(a1 + 32) = v21;
    v22 = v21;
    result = (unsigned int)*a9;
    if ( !(_DWORD)result )
    {
      if ( !(unsigned int)sub_8DD0E0(v22, &v49, &v50, &v51, v52)
        || !(v52[0] | (unsigned int)v51 | (unsigned int)v50 | v49) )
      {
        goto LABEL_22;
      }
      goto LABEL_26;
    }
LABEL_18:
    v15 |= a10[5].m128i_u32[2];
    goto LABEL_6;
  }
  if ( v19 != 1 )
  {
    v45 = *(_QWORD *)(a1 + 32);
    result = sub_8A4520(v45, (__int64)a5, a6, a7, a8, a9, (__int64)a10);
    if ( !*a9 )
    {
      *(_QWORD *)(a1 + 32) = result;
      if ( (((unsigned __int8)(a8 >> 10) ^ 1) & (v13 != 0)) == 0
        || v45 == result
        || (unsigned int)sub_8B2F00(a1, *(_QWORD *)(a2 + 64), a3, v13, 0, a7) )
      {
        goto LABEL_22;
      }
      goto LABEL_26;
    }
    goto LABEL_18;
  }
  if ( a4 )
  {
    v44 = *(__int64 ***)(*(_QWORD *)(a2 + 64) + 128LL);
    v50 = v44;
    if ( (*(_BYTE *)(a2 + 72) & 1) != 0 )
    {
      if ( (a8 & 0x408) != 0 )
      {
        v50 = sub_8A2270((__int64)v44, a3, a4, a7, a8 | 0x4000, a9, a10);
        result = (__int64)a9;
        if ( *a9 )
          goto LABEL_18;
        v20 = sub_8A2270((__int64)v50, a5, a6, a7, a8, a9, a10);
        v10 = a5;
        v50 = v20;
        result = (__int64)a9;
        v11 = a6;
        if ( *a9 )
          goto LABEL_18;
      }
      else
      {
        v50 = sub_8A2270((__int64)v44, a3, a4, a7, a8, a9, a10);
        result = (__int64)a9;
        v10 = a5;
        v11 = a6;
        if ( *a9 )
          goto LABEL_18;
      }
    }
    v13 = 0;
    if ( (*(_BYTE *)(a2 + 57) & 8) == 0 )
      v13 = v50;
  }
  else
  {
    v50 = 0;
    v44 = 0;
  }
  v23 = a8;
  LOBYTE(v23) = a8 | 0x80;
  v42 = v11;
  v47 = (__int64)v10;
  v24 = sub_744A50(*(__m128i **)(a1 + 32), v10, v11, v13, a7, v23, a9, a10);
  v25 = v38;
  *(_QWORD *)(a1 + 32) = v24;
  v26 = v24;
  result = (__int64)a9;
  if ( *a9 )
    goto LABEL_18;
  v27 = v50;
  if ( !v50 )
    goto LABEL_22;
  if ( (*(_BYTE *)(a2 + 57) & 8) != 0 )
  {
    if ( !(unsigned int)sub_696F90((__int64)v50, v26, 0, (__int64 *)&v50, 0, v47, v42) )
      goto LABEL_60;
    v28 = v50;
    v27 = v50;
    if ( v50 == v44 )
    {
LABEL_39:
      if ( (*(_BYTE *)(a2 + 57) & 8) != 0 )
        goto LABEL_22;
      v44 = v28;
      v26 = *(_QWORD *)(a1 + 32);
LABEL_41:
      v30 = v44;
      v31 = *(_QWORD *)(v26 + 128);
      v29 = *((unsigned __int8 *)v44 + 140);
      if ( (_BYTE)v29 == 12 )
      {
        do
        {
          v30 = (__int64 **)v30[20];
          v32 = *((_BYTE *)v30 + 140);
        }
        while ( v32 == 12 );
      }
      else
      {
        v32 = *((_BYTE *)v44 + 140);
      }
      if ( !v32 )
        goto LABEL_26;
      if ( *(_BYTE *)(v31 + 140) != 12 )
        goto LABEL_46;
      goto LABEL_45;
    }
  }
  else if ( v50 == v44 )
  {
    goto LABEL_41;
  }
  if ( (unsigned int)sub_8D2600(v27)
    || (unsigned int)sub_8D3A70(v50)
    && (dword_4F077C4 != 2 || unk_4F07778 <= 202001 || !(unsigned int)sub_8D42F0(v50, v26))
    || dword_4D04474 && (unsigned int)sub_8D3110(v50) )
  {
    goto LABEL_59;
  }
  v28 = v50;
  v29 = *((unsigned __int8 *)v50 + 140);
  v44 = v50;
  if ( (_BYTE)v29 != 3 )
    goto LABEL_39;
  if ( !dword_4D04800 )
  {
LABEL_59:
    sub_72C930();
LABEL_60:
    result = (__int64)a9;
    *a9 = 1;
    goto LABEL_18;
  }
  if ( (*(_BYTE *)(a2 + 57) & 8) != 0 )
    goto LABEL_22;
  v31 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 128LL);
  if ( *(_BYTE *)(v31 + 140) == 12 )
  {
    do
LABEL_45:
      v31 = *(_QWORD *)(v31 + 160);
    while ( *(_BYTE *)(v31 + 140) == 12 );
LABEL_46:
    if ( (_BYTE)v29 == 12 )
    {
      do
      {
        v33 = *((_BYTE *)v44[20] + 140) == 12;
        v44 = (__int64 **)v44[20];
      }
      while ( v33 );
    }
  }
  if ( !(unsigned int)sub_8D97D0(v44, v31, 0, v29, v25) )
  {
    v34 = *(const __m128i **)(a1 + 32);
    v35 = v50;
    if ( !(unsigned int)sub_8E1010(
                          v34[8].m128i_i64[0],
                          1,
                          0,
                          0,
                          0,
                          (_DWORD)v34,
                          (__int64)v50,
                          0,
                          0,
                          0,
                          0,
                          (__int64)v52,
                          0)
      || !(unsigned int)sub_8DD690(v52, v34[8].m128i_i64[0], 1, *(_QWORD *)(a1 + 32), v35, 0) )
    {
      goto LABEL_26;
    }
    v51 = (__m128i *)sub_724DC0();
    sub_724C70((__int64)v51, v34[10].m128i_i8[13]);
    sub_72A510(v34, v51);
    sub_7115B0(v51, (__int64)v35, 1, 1, 1, 1, 0, 0, 1u, 0, 0, &v48, &v49, a7);
    if ( v49 )
      v48 = 1;
    if ( v48 )
    {
      sub_724E30((__int64)&v51);
LABEL_26:
      *a9 = 1;
      goto LABEL_22;
    }
    sub_7296C0(&v49);
    v36 = sub_7401F0((__int64)v51);
    v37 = v49;
    *(_QWORD *)(a1 + 32) = v36;
    sub_729730(v37);
    sub_724E30((__int64)&v51);
  }
LABEL_22:
  result = *(unsigned __int8 *)(a1 + 24);
  if ( (result & 0x10) != 0 )
  {
    result = (16 * (a10[5].m128i_i32[2] & 1)) | (unsigned int)result & 0xFFFFFFEF;
    *(_BYTE *)(a1 + 24) = result;
  }
  else
  {
    v15 |= a10[5].m128i_u32[2];
  }
LABEL_6:
  a10[5].m128i_i32[2] = v15;
  return result;
}
