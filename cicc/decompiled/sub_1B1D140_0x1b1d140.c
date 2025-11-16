// Function: sub_1B1D140
// Address: 0x1b1d140
//
__int64 __fastcall sub_1B1D140(__int64 a1, __int64 a2, __int64 a3, __int64 a4, bool a5, __m128i a6, __m128i a7)
{
  char v8; // al
  _BYTE *v9; // r12
  __int64 result; // rax
  __int64 *v12; // r15
  __int64 *v13; // rax
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // rdi
  __int64 v17; // rax
  char v18; // r8
  unsigned int v19; // esi
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // r13
  __int64 v25; // r15
  __int64 v26; // r12
  __int64 *v27; // rsi
  int v28; // r8d
  int v29; // r9d
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r15
  __int64 v33; // [rsp+8h] [rbp-88h]
  __int64 v34; // [rsp+10h] [rbp-80h]
  __int64 v35; // [rsp+18h] [rbp-78h]
  _BYTE *v36; // [rsp+20h] [rbp-70h]
  bool v40; // [rsp+3Fh] [rbp-51h]
  unsigned __int8 v41; // [rsp+3Fh] [rbp-51h]
  _BYTE *v42; // [rsp+40h] [rbp-50h] BYREF
  __int64 v43; // [rsp+48h] [rbp-48h]
  _BYTE v44[64]; // [rsp+50h] [rbp-40h] BYREF

  v8 = *(_BYTE *)(*(_QWORD *)a1 + 8LL);
  v40 = (unsigned __int8)(v8 - 1) > 2u && (v8 & 0xFB) != 11;
  if ( v40 )
    return 0;
  v9 = (_BYTE *)a1;
  if ( (unsigned __int8)(v8 - 1) <= 5u )
    return sub_1B1B030(a1, a2, *(_QWORD *)(a3 + 112), a4);
  v12 = sub_1494E70(a3, a1, a6, a7);
  if ( *((_WORD *)v12 + 12) == 7 )
    return sub_1B16990(v9, a2, *(_QWORD *)(a3 + 112), a4, (__int64)v12, 0);
  if ( !a5 )
    return 0;
  v13 = sub_14951F0(a3, a1, a6, a7);
  v35 = (__int64)v13;
  if ( !v13 )
    return 0;
  if ( *((_WORD *)v12 + 12) != 10 )
    goto LABEL_49;
  if ( v12 == v13 )
    return sub_1B16990(v9, a2, *(_QWORD *)(a3 + 112), a4, (__int64)v12, 0);
  v14 = v13[6];
  v42 = v44;
  v43 = 0x200000000LL;
  v15 = *(v12 - 1);
  v34 = v14;
  v16 = sub_13FCB50(v14);
  if ( !v16 )
    goto LABEL_47;
  v17 = 0x17FFFFFFE8LL;
  v18 = *(_BYTE *)(v15 + 23) & 0x40;
  v19 = *(_DWORD *)(v15 + 20) & 0xFFFFFFF;
  if ( v19 )
  {
    v20 = 24LL * *(unsigned int *)(v15 + 56) + 8;
    v21 = 0;
    do
    {
      v22 = v15 - 24LL * v19;
      if ( v18 )
        v22 = *(_QWORD *)(v15 - 8);
      if ( v16 == *(_QWORD *)(v22 + v20) )
      {
        v17 = 24 * v21;
        goto LABEL_20;
      }
      ++v21;
      v20 += 8;
    }
    while ( v19 != (_DWORD)v21 );
    v17 = 0x17FFFFFFE8LL;
  }
LABEL_20:
  v23 = v18 ? *(_QWORD *)(v15 - 8) : v15 - 24LL * v19;
  v24 = *(_QWORD *)(v23 + v17);
  if ( !v24 )
    goto LABEL_47;
  v36 = v9;
  v25 = 0;
  if ( *(_BYTE *)(v24 + 16) > 0x17u )
    v25 = *(_QWORD *)(v23 + v17);
  v26 = v25;
  if ( v15 == v24 )
  {
LABEL_40:
    v9 = v36;
  }
  else
  {
    while ( 1 )
    {
      if ( !v26 || !sub_1377F70(v34 + 56, *(_QWORD *)(v26 + 40)) )
      {
LABEL_46:
        v9 = v36;
        goto LABEL_47;
      }
      if ( (v27 = sub_1494E70(a3, v24, a6, a7), *((_WORD *)v27 + 12) == 7) && sub_1478D00(a3, v27, v35) || v40 )
      {
        v30 = (unsigned int)v43;
        if ( (_DWORD)v43 )
        {
          v31 = *(_QWORD *)(v26 + 8);
          if ( !v31 || *(_QWORD *)(v31 + 8) )
            goto LABEL_46;
        }
        if ( (unsigned int)v43 >= HIDWORD(v43) )
        {
          sub_16CD150((__int64)&v42, v44, 0, 8, v28, v29);
          v30 = (unsigned int)v43;
        }
        *(_QWORD *)&v42[8 * v30] = v26;
        LODWORD(v43) = v43 + 1;
        v40 = a5;
        if ( (unsigned __int8)(*(_BYTE *)(v24 + 16) - 35) > 0x11u )
          goto LABEL_46;
      }
      else if ( (unsigned __int8)(*(_BYTE *)(v24 + 16) - 35) > 0x11u )
      {
        goto LABEL_46;
      }
      v26 = *(_QWORD *)(v24 - 24);
      v33 = *(_QWORD *)(v24 - 48);
      if ( !sub_13FC1A0(v34, v33) )
      {
        if ( !sub_13FC1A0(v34, v26) )
          goto LABEL_46;
        v26 = v33;
      }
      if ( !v26 )
        goto LABEL_46;
      if ( *(_BYTE *)(v26 + 16) <= 0x17u )
        break;
      v24 = v26;
      if ( v15 == v26 )
        goto LABEL_40;
    }
    v32 = v26;
    v9 = v36;
    if ( v15 != v32 )
      goto LABEL_47;
  }
  if ( !v40 )
  {
LABEL_47:
    if ( v42 != v44 )
      _libc_free((unsigned __int64)v42);
LABEL_49:
    v12 = (__int64 *)v35;
    return sub_1B16990(v9, a2, *(_QWORD *)(a3 + 112), a4, (__int64)v12, 0);
  }
  result = sub_1B16990(v9, a2, *(_QWORD *)(a3 + 112), a4, v35, (__int64)&v42);
  if ( v42 != v44 )
  {
    v41 = result;
    _libc_free((unsigned __int64)v42);
    return v41;
  }
  return result;
}
