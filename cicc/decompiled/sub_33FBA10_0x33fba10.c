// Function: sub_33FBA10
// Address: 0x33fba10
//
unsigned __int8 *__fastcall sub_33FBA10(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int64 a7,
        __int64 a8)
{
  int v8; // r10d
  int v9; // r12d
  __int64 v11; // r11
  unsigned __int8 *result; // rax
  __int16 v13; // ax
  __m128i *v14; // rax
  int v15; // r10d
  __int64 v16; // r11
  __int64 v17; // rdx
  unsigned __int8 *v18; // rax
  __int16 v19; // ax
  __int64 v20; // rsi
  int v21; // r14d
  unsigned __int64 v22; // rbx
  unsigned __int64 v23; // r9
  int v24; // r8d
  unsigned __int8 *v25; // rsi
  int v26; // r10d
  __int64 v27; // rcx
  unsigned __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // [rsp+0h] [rbp-110h]
  int v31; // [rsp+Ch] [rbp-104h]
  int v32; // [rsp+Ch] [rbp-104h]
  unsigned __int64 *v33; // [rsp+10h] [rbp-100h]
  __int64 v35; // [rsp+10h] [rbp-100h]
  __int64 v37; // [rsp+10h] [rbp-100h]
  int v38; // [rsp+10h] [rbp-100h]
  int v40; // [rsp+18h] [rbp-F8h]
  unsigned __int8 *v41; // [rsp+18h] [rbp-F8h]
  int v43; // [rsp+18h] [rbp-F8h]
  int v44; // [rsp+18h] [rbp-F8h]
  int v45; // [rsp+18h] [rbp-F8h]
  unsigned __int64 v46; // [rsp+18h] [rbp-F8h]
  __int64 v47; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v48; // [rsp+28h] [rbp-E8h]
  __int64 *v49; // [rsp+38h] [rbp-D8h] BYREF
  unsigned __int64 v50; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v51; // [rsp+48h] [rbp-C8h]
  __int64 v52[2]; // [rsp+50h] [rbp-C0h] BYREF
  _BYTE v53[176]; // [rsp+60h] [rbp-B0h] BYREF

  v8 = a6;
  v9 = a2;
  v11 = a7;
  v47 = a4;
  v48 = a5;
  if ( (_DWORD)a8 == 2 )
    return (unsigned __int8 *)sub_3405C90((_DWORD)a1, a2, a3, v47, a5, a6, *(_OWORD *)a7, *(_OWORD *)(a7 + 16));
  if ( (unsigned int)a8 <= 2 )
  {
    if ( (_DWORD)a8 )
      return sub_33FA050((__int64)a1, a2, a3, (unsigned int)v47, a5, a6, *(unsigned __int8 **)a7, *(_QWORD *)(a7 + 8));
    else
      return (unsigned __int8 *)sub_33F17F0(a1, a2, a3, v47, a5);
  }
  if ( (_DWORD)a8 == 3 )
    return (unsigned __int8 *)sub_340EC60(
                                (_DWORD)a1,
                                a2,
                                a3,
                                v47,
                                a5,
                                a6,
                                *(_QWORD *)a7,
                                *(_QWORD *)(a7 + 8),
                                *(_OWORD *)(a7 + 16),
                                *(_OWORD *)(a7 + 32));
  if ( (unsigned int)a2 <= 0x1DF )
  {
    if ( (unsigned int)a2 > 0x1D6 )
    {
      switch ( (int)a2 )
      {
        case 471:
          if ( (_WORD)v47 == 2 )
            v9 = 475;
          break;
        case 472:
        case 476:
        case 479:
          if ( (_WORD)v47 == 2 )
            v9 = 473;
          break;
        case 477:
        case 478:
          if ( (_WORD)v47 == 2 )
            v9 = 474;
          break;
        default:
          break;
      }
    }
    else
    {
      if ( (_DWORD)a2 == 395 )
      {
LABEL_26:
        v19 = sub_3281100((unsigned __int16 *)&v47, a2);
        v8 = a6;
        v11 = a7;
        if ( v19 == 2 )
          v9 = 407;
        goto LABEL_16;
      }
      if ( (unsigned int)a2 > 0x18B )
      {
        if ( (_DWORD)a2 == 399 )
        {
          v13 = sub_3281100((unsigned __int16 *)&v47, a2);
          v8 = a6;
          v11 = a7;
          if ( v13 == 2 )
            v9 = 396;
          goto LABEL_16;
        }
        if ( (_DWORD)a2 != 404 )
          goto LABEL_16;
        goto LABEL_26;
      }
      if ( (_DWORD)a2 == 156 )
      {
        result = (unsigned __int8 *)sub_33F2070(v47, a5, (char *)a7, a8, a1);
        v11 = a7;
        v8 = a6;
        if ( result )
          return result;
      }
      else if ( (_DWORD)a2 == 159 )
      {
        result = (unsigned __int8 *)sub_33FC250(a3, (unsigned int)v47, a5, a7, a8, a1);
        v11 = a7;
        v8 = a6;
        if ( result )
          return result;
      }
    }
  }
LABEL_16:
  v33 = (unsigned __int64 *)v11;
  v40 = v8;
  v14 = sub_33ED250((__int64)a1, (unsigned int)v47, v48);
  v15 = v40;
  v50 = (unsigned __int64)v14;
  v16 = (__int64)v33;
  v51 = v17;
  if ( (_WORD)v47 == 262 )
  {
    v20 = *(_QWORD *)a3;
    v21 = *(_DWORD *)(a3 + 8);
    v52[0] = v20;
    if ( v20 )
    {
      sub_B96E90((__int64)v52, v20, 1);
      v16 = (__int64)v33;
      v15 = v40;
    }
    v22 = a1[52];
    v23 = v50;
    v24 = v51;
    if ( v22 )
    {
      a1[52] = *(_QWORD *)v22;
    }
    else
    {
      v27 = a1[53];
      a1[63] += 120LL;
      v28 = (v27 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      if ( a1[54] >= v28 + 120 && v27 )
      {
        a1[53] = v28 + 120;
        if ( !v28 )
        {
          if ( v52[0] )
          {
            v37 = v16;
            v45 = v15;
            sub_B91220((__int64)v52, v52[0]);
            v16 = v37;
            v15 = v45;
          }
          goto LABEL_37;
        }
        v22 = (v27 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      }
      else
      {
        v30 = v16;
        v32 = v15;
        v38 = v24;
        v46 = v23;
        v29 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
        v16 = v30;
        v15 = v32;
        v24 = v38;
        v23 = v46;
        v22 = v29;
      }
    }
    *(_QWORD *)v22 = 0;
    *(_QWORD *)(v22 + 8) = 0;
    *(_QWORD *)(v22 + 16) = 0;
    *(_DWORD *)(v22 + 24) = v9;
    *(_DWORD *)(v22 + 28) = 0;
    *(_WORD *)(v22 + 34) = -1;
    *(_DWORD *)(v22 + 36) = -1;
    *(_QWORD *)(v22 + 40) = 0;
    *(_QWORD *)(v22 + 48) = v23;
    *(_QWORD *)(v22 + 56) = 0;
    *(_DWORD *)(v22 + 64) = 0;
    *(_DWORD *)(v22 + 68) = v24;
    *(_DWORD *)(v22 + 72) = v21;
    v25 = (unsigned __int8 *)v52[0];
    *(_QWORD *)(v22 + 80) = v52[0];
    if ( v25 )
    {
      v35 = v16;
      v43 = v15;
      sub_B976B0((__int64)v52, v25, v22 + 80);
      v16 = v35;
      v15 = v43;
    }
    *(_QWORD *)(v22 + 88) = 0xFFFFFFFFLL;
    *(_WORD *)(v22 + 32) = 0;
LABEL_37:
    v44 = v15;
    sub_33E4EC0((__int64)a1, v22, v16, a8);
    v26 = v44;
LABEL_38:
    *(_DWORD *)(v22 + 28) = v26;
    sub_33CC420((__int64)a1, v22);
    return (unsigned __int8 *)v22;
  }
  v31 = v40;
  v52[1] = 0x2000000000LL;
  v52[0] = (__int64)v53;
  sub_33C9670((__int64)v52, v9, v50, v33, a8, (__int64)v52);
  v49 = 0;
  v18 = (unsigned __int8 *)sub_33CCCF0((__int64)a1, (__int64)v52, a3, (__int64 *)&v49);
  if ( !v18 )
  {
    v22 = sub_33E6540(a1, v9, *(_DWORD *)(a3 + 8), (__int64 *)a3, (__int64 *)&v50);
    sub_33E4EC0((__int64)a1, v22, (__int64)v33, a8);
    sub_C657C0(a1 + 65, (__int64 *)v22, v49, (__int64)off_4A367D0);
    v26 = v40;
    if ( (_BYTE *)v52[0] != v53 )
    {
      _libc_free(v52[0]);
      v26 = v40;
    }
    goto LABEL_38;
  }
  v41 = v18;
  sub_33D00A0((__int64)v18, v31);
  result = v41;
  if ( (_BYTE *)v52[0] != v53 )
  {
    _libc_free(v52[0]);
    return v41;
  }
  return result;
}
