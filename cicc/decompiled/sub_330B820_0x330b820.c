// Function: sub_330B820
// Address: 0x330b820
//
__int64 __fastcall sub_330B820(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        int a10,
        unsigned int a11,
        char a12)
{
  __int64 result; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int32 v17; // r11d
  int v18; // edx
  __int64 v19; // rax
  int v20; // ecx
  __int64 v21; // rax
  __int8 v22; // cl
  __int64 (*v23)(); // rax
  __int8 v24; // al
  __int64 v25; // rsi
  __int64 v26; // r11
  __int64 *v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __m128i v30; // rax
  __int64 v31; // rax
  int v32; // ebx
  bool v33; // al
  int v34; // r9d
  __int64 v35; // rsi
  __int64 v36; // rbx
  int v37; // ecx
  __int64 v38; // rbx
  __int32 v39; // edx
  __int32 v40; // r14d
  __int64 v41; // [rsp+0h] [rbp-D0h]
  __int64 v42; // [rsp+8h] [rbp-C8h]
  __int64 v43; // [rsp+18h] [rbp-B8h]
  int v44; // [rsp+18h] [rbp-B8h]
  __m128i v45; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v46; // [rsp+30h] [rbp-A0h]
  _BYTE *v47; // [rsp+38h] [rbp-98h]
  __int64 v48; // [rsp+40h] [rbp-90h] BYREF
  __int64 v49; // [rsp+48h] [rbp-88h]
  __m128i v50; // [rsp+50h] [rbp-80h] BYREF
  __int64 v51; // [rsp+60h] [rbp-70h]
  int v52; // [rsp+68h] [rbp-68h]
  unsigned __int64 v53[2]; // [rsp+70h] [rbp-60h] BYREF
  _BYTE v54[80]; // [rsp+80h] [rbp-50h] BYREF

  v48 = a4;
  v49 = a5;
  if ( *(_DWORD *)(a8 + 24) != 298 || (*(_BYTE *)(a8 + 33) & 0xC) != 0 || (*(_WORD *)(a8 + 32) & 0x380) != 0 )
    return 0;
  if ( a12 )
  {
    v15 = *(_QWORD *)(a8 + 56);
    if ( v15 )
    {
      while ( 1 )
      {
        v16 = *(_QWORD *)(v15 + 16);
        if ( *(_DWORD *)(v16 + 24) == 208
          && (unsigned int)(*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v16 + 40) + 80LL) + 96LL) - 18) <= 3 )
        {
          break;
        }
        v15 = *(_QWORD *)(v15 + 32);
        if ( !v15 )
          goto LABEL_13;
      }
      a11 = 213;
      a10 = 2;
    }
  }
LABEL_13:
  v17 = (unsigned __int16)v48;
  v18 = a9;
  if ( !a6 )
  {
    if ( (_WORD)v48 )
    {
      if ( (unsigned __int16)(v48 - 17) <= 0x9Eu || (*(_BYTE *)(*(_QWORD *)(a8 + 112) + 37LL) & 0xF) != 0 )
        goto LABEL_28;
    }
    else
    {
      v45.m128i_i32[0] = (unsigned __int16)v48;
      LODWORD(v47) = a9;
      if ( sub_30070D0((__int64)&v48) )
        return 0;
      v18 = (int)v47;
      v17 = v45.m128i_i32[0];
      if ( (*(_BYTE *)(*(_QWORD *)(a8 + 112) + 37LL) & 0xF) != 0 )
        return 0;
    }
    if ( (*(_BYTE *)(a8 + 32) & 8) == 0 )
      goto LABEL_18;
  }
LABEL_28:
  v21 = *(unsigned __int16 *)(*(_QWORD *)(a8 + 48) + 16LL * (unsigned int)a9);
  if ( !(_WORD)v17
    || !(_WORD)v21
    || (((int)*(unsigned __int16 *)(a3 + 2 * (v21 + 274LL * (unsigned __int16)v17 + 71704) + 6) >> (4 * a10)) & 0xF) != 0 )
  {
    return 0;
  }
LABEL_18:
  v47 = v54;
  v53[0] = (unsigned __int64)v54;
  v53[1] = 0x400000000LL;
  v19 = *(_QWORD *)(a8 + 56);
  if ( v19 )
  {
    v20 = 1;
    do
    {
      if ( *(_DWORD *)(v19 + 8) == v18 )
      {
        if ( !v20 )
          goto LABEL_40;
        v19 = *(_QWORD *)(v19 + 32);
        if ( !v19 )
        {
          v22 = 1;
          goto LABEL_41;
        }
        if ( v18 == *(_DWORD *)(v19 + 8) )
          goto LABEL_40;
        v20 = 0;
      }
      v19 = *(_QWORD *)(v19 + 32);
    }
    while ( v19 );
    if ( v20 == 1 )
      goto LABEL_40;
    if ( (_WORD)v17 )
    {
      if ( (unsigned __int16)(v17 - 17) > 0xD3u )
        goto LABEL_44;
      v22 = 1;
    }
    else
    {
      v22 = sub_30070B0((__int64)&v48);
      if ( !v22 )
        goto LABEL_44;
    }
  }
  else
  {
LABEL_40:
    v45.m128i_i32[0] = v17;
    v24 = sub_32611B0(v48, v49, a7, a8, a9, a11, (__int64)v53, a3);
    LOWORD(v17) = v45.m128i_i16[0];
    v22 = v24;
LABEL_41:
    if ( (_WORD)v17 )
    {
      if ( (unsigned __int16)(v17 - 17) > 0xD3u )
        goto LABEL_43;
    }
    else
    {
      v45.m128i_i8[0] = v22;
      v33 = sub_30070B0((__int64)&v48);
      v22 = v45.m128i_i8[0];
      if ( !v33 )
        goto LABEL_43;
    }
  }
  v23 = *(__int64 (**)())(*(_QWORD *)a3 + 1584LL);
  if ( v23 != sub_2FE3520 )
  {
    v45.m128i_i8[0] = v22;
    v22 &= ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v23)(a3, a7, 0);
LABEL_43:
    if ( !v22 )
      goto LABEL_37;
LABEL_44:
    v25 = *(_QWORD *)(a8 + 80);
    v26 = *(_QWORD *)(a8 + 112);
    v27 = *(__int64 **)(a8 + 40);
    v43 = 16LL * (unsigned int)a9;
    v28 = *(unsigned __int16 *)(*(_QWORD *)(a8 + 48) + v43);
    v29 = *(_QWORD *)(*(_QWORD *)(a8 + 48) + v43 + 8);
    v50.m128i_i64[0] = v25;
    if ( v25 )
    {
      v41 = v28;
      v42 = v29;
      v46 = v26;
      v45.m128i_i64[0] = (__int64)v27;
      sub_B96E90((__int64)&v50, v25, 1);
      v28 = v41;
      v29 = v42;
      LODWORD(v26) = v46;
      v27 = (__int64 *)v45.m128i_i64[0];
    }
    v50.m128i_i32[2] = *(_DWORD *)(a8 + 72);
    v30.m128i_i64[0] = sub_33F1B30(a1, a10, (unsigned int)&v50, v48, v49, v26, *v27, v27[1], v27[5], v27[6], v28, v29);
    v45 = v30;
    v46 = v30.m128i_i64[0];
    if ( v50.m128i_i64[0] )
      sub_B91220((__int64)&v50, v50.m128i_i64[0]);
    sub_3304760(a2, (__int64)v53, a8, a9, v45.m128i_i64[0], v45.m128i_i64[1], a11);
    v31 = *(_QWORD *)(a8 + 56);
    if ( v31 )
    {
      v32 = 1;
      do
      {
        if ( !*(_DWORD *)(v31 + 8) )
        {
          if ( !v32 )
            goto LABEL_61;
          v31 = *(_QWORD *)(v31 + 32);
          if ( !v31 )
          {
            v50 = _mm_load_si128(&v45);
            sub_32EB790((__int64)a2, a7, v50.m128i_i64, 1, 1);
            goto LABEL_59;
          }
          if ( !*(_DWORD *)(v31 + 8) )
            goto LABEL_61;
          v32 = 0;
        }
        v31 = *(_QWORD *)(v31 + 32);
      }
      while ( v31 );
      v50 = _mm_load_si128(&v45);
      sub_32EB790((__int64)a2, a7, v50.m128i_i64, 1, 1);
      if ( v32 == 1 )
        goto LABEL_62;
LABEL_59:
      sub_34161C0(a1, a8, 1, v46, 1);
      sub_32CF870((__int64)a2, a8);
    }
    else
    {
LABEL_61:
      v50 = _mm_load_si128(&v45);
      sub_32EB790((__int64)a2, a7, v50.m128i_i64, 1, 1);
LABEL_62:
      v35 = *(_QWORD *)(a8 + 80);
      v36 = *(_QWORD *)(*(_QWORD *)(a8 + 48) + v43 + 8);
      v37 = *(unsigned __int16 *)(*(_QWORD *)(a8 + 48) + 16LL * (unsigned int)a9);
      v50.m128i_i64[0] = v35;
      if ( v35 )
      {
        v44 = v37;
        sub_B96E90((__int64)&v50, v35, 1);
        v37 = v44;
      }
      v50.m128i_i32[2] = *(_DWORD *)(a8 + 72);
      v38 = sub_33FAF80(a1, 216, (unsigned int)&v50, v37, v36, v34, *(_OWORD *)&v45);
      v40 = v39;
      if ( v50.m128i_i64[0] )
        sub_B91220((__int64)&v50, v50.m128i_i64[0]);
      v50.m128i_i64[0] = v38;
      v50.m128i_i32[2] = v40;
      v51 = v46;
      v52 = 1;
      sub_32EB790((__int64)a2, a8, v50.m128i_i64, 2, 1);
    }
    result = a7;
    goto LABEL_38;
  }
LABEL_37:
  result = 0;
LABEL_38:
  if ( (_BYTE *)v53[0] != v47 )
  {
    v45.m128i_i64[0] = 0;
    v47 = (_BYTE *)result;
    _libc_free(v53[0]);
    return (__int64)v47;
  }
  return result;
}
