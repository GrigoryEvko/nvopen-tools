// Function: sub_199D980
// Address: 0x199d980
//
__int64 __fastcall sub_199D980(__int64 a1, __int64 a2, __m128i a3, __m128i a4)
{
  _QWORD *v6; // r13
  __int16 v7; // ax
  __int64 v8; // r15
  unsigned int v9; // esi
  unsigned __int64 v10; // rdx
  __int64 v11; // rax
  unsigned int v12; // eax
  __int64 v13; // r14
  __int64 v15; // rdi
  int v16; // ecx
  _BYTE *v17; // rsi
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rcx
  __int64 v21; // rax
  unsigned int v22; // edx
  _BYTE *v23; // rsi
  _BYTE *v24; // rdx
  __int64 *v25; // [rsp+10h] [rbp-80h] BYREF
  __int64 v26; // [rsp+18h] [rbp-78h]
  _BYTE v27[112]; // [rsp+20h] [rbp-70h] BYREF

  v6 = *(_QWORD **)a1;
  v7 = *(_WORD *)(*(_QWORD *)a1 + 24LL);
  if ( !v7 )
  {
    v8 = v6[4];
    v9 = *(_DWORD *)(v8 + 32);
    v10 = *(_QWORD *)(v8 + 24);
    v11 = 1LL << ((unsigned __int8)v9 - 1);
    if ( v9 > 0x40 )
    {
      v15 = v8 + 24;
      if ( (*(_QWORD *)(v10 + 8LL * ((v9 - 1) >> 6)) & v11) == 0 )
      {
        v12 = v9 + 1 - sub_16A57B0(v15);
LABEL_6:
        v13 = 0;
        if ( v12 > 0x40 )
          return v13;
        goto LABEL_18;
      }
      v16 = sub_16A5810(v15);
    }
    else
    {
      if ( (v11 & v10) == 0 )
      {
        if ( v10 )
        {
          _BitScanReverse64(&v10, v10);
          v12 = 65 - (v10 ^ 0x3F);
          goto LABEL_6;
        }
LABEL_18:
        *(_QWORD *)a1 = sub_145CF80(a2, *(_QWORD *)v8, 0, 0);
        v21 = v6[4];
        v22 = *(_DWORD *)(v21 + 32);
        if ( v22 > 0x40 )
          return **(_QWORD **)(v21 + 24);
        else
          return (__int64)(*(_QWORD *)(v21 + 24) << (64 - (unsigned __int8)v22)) >> (64 - (unsigned __int8)v22);
      }
      v16 = 64;
      v19 = ~(v10 << (64 - (unsigned __int8)v9));
      if ( v19 )
      {
        _BitScanReverse64(&v20, v19);
        v16 = v20 ^ 0x3F;
      }
    }
    v13 = 0;
    if ( v9 + 1 - v16 > 0x40 )
      return v13;
    goto LABEL_18;
  }
  if ( v7 == 4 )
  {
    v17 = (_BYTE *)v6[4];
    v18 = v6[5];
    v25 = (__int64 *)v27;
    v26 = 0x800000000LL;
    sub_145C5B0((__int64)&v25, v17, &v17[8 * v18]);
    v13 = sub_199D980(v25, a2);
    if ( v13 )
      *(_QWORD *)a1 = sub_147DD40(a2, (__int64 *)&v25, 0, 0, a3, a4);
  }
  else
  {
    v13 = 0;
    if ( v7 != 7 )
      return v13;
    v23 = (_BYTE *)v6[4];
    v24 = &v23[8 * v6[5]];
    v25 = (__int64 *)v27;
    v26 = 0x800000000LL;
    sub_145C5B0((__int64)&v25, v23, v24);
    v13 = sub_199D980(v25, a2);
    if ( v13 )
      *(_QWORD *)a1 = sub_14785F0(a2, &v25, v6[6], 0);
  }
  if ( v25 != (__int64 *)v27 )
    _libc_free((unsigned __int64)v25);
  return v13;
}
