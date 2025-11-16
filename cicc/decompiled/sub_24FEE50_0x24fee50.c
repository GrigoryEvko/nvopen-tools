// Function: sub_24FEE50
// Address: 0x24fee50
//
__int64 __fastcall sub_24FEE50(__int64 a1, unsigned __int8 *a2, __int64 a3, char a4)
{
  __int64 v7; // r15
  __int64 result; // rax
  unsigned int v9; // edx
  __int64 v10; // rsi
  unsigned __int64 v11; // rax
  unsigned int v12; // r15d
  unsigned int v13; // r15d
  int v14; // eax
  unsigned __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rdx
  __int64 v18; // rcx
  unsigned __int16 v19; // cx
  __int64 v20; // rsi
  unsigned __int64 v21; // rax
  __int8 v22; // dl
  unsigned __int8 *v23; // rax
  __m128i *v24; // r8
  unsigned int v25; // eax
  __int64 v26; // rdx
  __int64 v27; // rax
  unsigned __int64 v28; // rcx
  unsigned __int64 v29; // rax
  __m128i *v30; // rcx
  __m128i *v31; // rax
  unsigned __int64 v32; // rax
  unsigned __int8 v33; // di
  __m128i *v34; // rax
  unsigned __int8 v35; // cl
  unsigned int v36; // [rsp+0h] [rbp-B0h]
  unsigned int v38; // [rsp+Ch] [rbp-A4h]
  __int64 v39; // [rsp+18h] [rbp-98h] BYREF
  unsigned __int64 v40; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v41; // [rsp+28h] [rbp-88h]
  __m128i v42; // [rsp+30h] [rbp-80h] BYREF
  unsigned __int8 *v43; // [rsp+40h] [rbp-70h]
  _QWORD v44[2]; // [rsp+50h] [rbp-60h] BYREF
  __int64 v45; // [rsp+60h] [rbp-50h]
  char v46; // [rsp+70h] [rbp-40h]

  if ( sub_B46500(a2) || (a2[2] & 1) != 0 )
    return 256;
  v7 = *((_QWORD *)a2 - 4);
  v41 = sub_AE43F0(*(_QWORD *)a1, *(_QWORD *)(v7 + 8));
  if ( v41 > 0x40 )
    sub_C43690((__int64)&v40, 0, 0);
  else
    v40 = 0;
  if ( **(unsigned __int8 ***)(a1 + 8) != sub_BD45C0((unsigned __int8 *)v7, *(_QWORD *)a1, (__int64)&v40, 1, 0, 0, 0, 0) )
  {
    result = 0;
    v9 = v41;
    goto LABEL_7;
  }
  v9 = v41;
  v10 = 1LL << ((unsigned __int8)v41 - 1);
  if ( v41 > 0x40 )
  {
    v13 = v41 + 1;
    v36 = v41;
    if ( (*(_QWORD *)(v40 + 8LL * ((v41 - 1) >> 6)) & v10) != 0 )
      v14 = sub_C44500((__int64)&v40);
    else
      v14 = sub_C444A0((__int64)&v40);
    v9 = v36;
    v12 = v13 - v14;
  }
  else if ( (v10 & v40) != 0 )
  {
    if ( !v41 )
      goto LABEL_26;
    if ( v40 << (64 - (unsigned __int8)v41) == -1 )
    {
      v12 = v41 - 63;
    }
    else
    {
      _BitScanReverse64(&v11, ~(v40 << (64 - (unsigned __int8)v41)));
      v12 = v41 + 1 - (v11 ^ 0x3F);
    }
  }
  else
  {
    if ( !v40 )
      goto LABEL_26;
    _BitScanReverse64(&v15, v40);
    v12 = 65 - (v15 ^ 0x3F);
  }
  if ( v12 > 0x3F )
  {
LABEL_22:
    result = 256;
    goto LABEL_7;
  }
LABEL_26:
  v44[0] = sub_9208B0(*(_QWORD *)a1, a3);
  v16 = v44[0];
  v44[1] = v17;
  if ( !(_BYTE)v17 )
  {
    v9 = v41;
    if ( **(_BYTE **)(a1 + 16) && *(_BYTE *)(a3 + 8) == 14 )
      goto LABEL_22;
    if ( v41 > 0x40 )
    {
      v18 = *(_QWORD *)v40;
    }
    else
    {
      v18 = 0;
      if ( v41 )
        v18 = (__int64)(v40 << (64 - (unsigned __int8)v41)) >> (64 - (unsigned __int8)v41);
    }
    v39 = v18;
    v19 = *((_WORD *)a2 + 1);
    v20 = *(_QWORD *)(a1 + 24);
    v42.m128i_i64[0] = a3;
    _BitScanReverse64(&v21, 1LL << (v19 >> 1));
    v22 = 63 - (v21 ^ 0x3F);
    v23 = 0;
    if ( a4 )
      v23 = a2;
    v42.m128i_i8[8] = v22;
    v43 = v23;
    sub_24FEA80((__int64)v44, v20, &v39, &v42);
    v24 = &v42;
    v25 = **(_DWORD **)(a1 + 32);
    if ( !v25 || v25 >= *(_DWORD *)(*(_QWORD *)(a1 + 24) + 8LL) >> 1 )
    {
      v26 = v45;
      if ( *(_QWORD *)(v45 + 8) == a3 )
      {
        if ( !a4 )
        {
          if ( !v46 )
          {
            LOWORD(v27) = *((_WORD *)a2 + 1);
            _BitScanReverse64(&v28, 1LL << ((unsigned __int16)v27 >> 1));
            if ( *(_BYTE *)(v45 + 16) >= (unsigned __int8)(63 - (v28 ^ 0x3F)) )
              goto LABEL_51;
          }
          if ( v39 < 0 )
            goto LABEL_37;
          _BitScanReverse64(&v29, 1LL << (*((_WORD *)a2 + 1) >> 1));
          if ( (v39 & ~(-1LL << (63 - ((unsigned __int8)v29 ^ 0x3Fu)))) != 0 )
            goto LABEL_37;
          v30 = *(__m128i **)(a1 + 40);
          v42.m128i_i64[0] = v39 + ((unsigned __int64)(v16 + 7) >> 3);
          v31 = &v42;
          if ( v42.m128i_i64[0] <= (unsigned __int64)v30->m128i_i64[0] )
            v31 = v30;
          v30->m128i_i64[0] = v31->m128i_i64[0];
          _BitScanReverse64(&v32, 1LL << (*((_WORD *)a2 + 1) >> 1));
          v33 = 63 - (v32 ^ 0x3F);
          v34 = *(__m128i **)(a1 + 48);
          v42.m128i_i8[0] = v33;
          if ( (unsigned int)v34->m128i_i8[0] >= v33 )
            v24 = v34;
          v34->m128i_i8[0] = v24->m128i_i8[0];
        }
        LOWORD(v27) = *((_WORD *)a2 + 1);
LABEL_51:
        _BitScanReverse64((unsigned __int64 *)&v27, 1LL << ((unsigned __int16)v27 >> 1));
        v35 = 63 - (v27 ^ 0x3F);
        if ( *(_BYTE *)(v26 + 16) < v35 )
          *(_BYTE *)(v26 + 16) = v35;
        v9 = v41;
        result = 257;
        goto LABEL_7;
      }
    }
  }
LABEL_37:
  v9 = v41;
  result = 256;
LABEL_7:
  if ( v9 > 0x40 )
  {
    if ( v40 )
    {
      v38 = result;
      j_j___libc_free_0_0(v40);
      return v38;
    }
  }
  return result;
}
