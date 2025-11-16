// Function: sub_1D24DC0
// Address: 0x1d24dc0
//
__int64 __fastcall sub_1D24DC0(
        _QWORD *a1,
        unsigned __int16 a2,
        __int64 a3,
        __int64 a4,
        int a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8,
        unsigned __int8 a9,
        __int64 a10)
{
  __int64 *v11; // r15
  __int64 *v12; // rbx
  __int64 v13; // rsi
  int v14; // r11d
  unsigned __int16 v15; // r15
  int v16; // eax
  _QWORD *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  _QWORD *v22; // rbx
  __int64 result; // rax
  __int64 v24; // rbx
  int v25; // r12d
  int v26; // r12d
  __int128 v27; // [rsp-20h] [rbp-1B0h]
  __int128 v28; // [rsp-20h] [rbp-1B0h]
  __int128 v29; // [rsp-20h] [rbp-1B0h]
  unsigned __int8 *v34; // [rsp+58h] [rbp-138h] BYREF
  __int64 *v35[3]; // [rsp+60h] [rbp-130h] BYREF
  char v36; // [rsp+7Ah] [rbp-116h]
  char v37; // [rsp+7Bh] [rbp-115h]
  __int64 v38[5]; // [rsp+A8h] [rbp-E8h] BYREF
  unsigned __int64 v39[2]; // [rsp+D0h] [rbp-C0h] BYREF
  _BYTE v40[176]; // [rsp+E0h] [rbp-B0h] BYREF

  v11 = a7;
  if ( *(_BYTE *)(a4 + 16LL * (unsigned int)(a5 - 1)) == 111 )
  {
    v24 = a1[26];
    v25 = *(_DWORD *)(a3 + 8);
    if ( v24 )
      a1[26] = *(_QWORD *)v24;
    else
      v24 = sub_145CBF0(a1 + 27, 112, 8);
    *((_QWORD *)&v28 + 1) = a10;
    *(_QWORD *)&v28 = a9;
    sub_1D189E0(v24, a2, v25, (unsigned __int8 **)a3, a4, a5, v28, a6);
    *(_BYTE *)(v24 + 26) |= 2u;
    sub_1D23B60((__int64)a1, v24, (__int64)a7, a8);
    goto LABEL_14;
  }
  v12 = &a7[2 * a8];
  v39[0] = (unsigned __int64)v40;
  v39[1] = 0x2000000000LL;
  sub_16BD430((__int64)v39, a2);
  sub_16BD4C0((__int64)v39, a4);
  if ( v12 != a7 )
  {
    do
    {
      v13 = *v11;
      v11 += 2;
      sub_16BD4C0((__int64)v39, v13);
      sub_16BD430((__int64)v39, *((_DWORD *)v11 - 2));
    }
    while ( v12 != v11 );
  }
  v14 = *(_DWORD *)(a3 + 8);
  *((_QWORD *)&v27 + 1) = a10;
  *(_QWORD *)&v27 = a9;
  v34 = 0;
  sub_1D189E0((__int64)v35, a2, v14, &v34, a4, a5, v27, a6);
  v36 |= 2u;
  HIBYTE(v15) = v37;
  LOBYTE(v15) = v36 & 0xFA;
  if ( v38[0] )
    sub_161E7C0((__int64)v38, v38[0]);
  if ( v34 )
    sub_161E7C0((__int64)&v34, (__int64)v34);
  sub_16BD3E0((__int64)v39, v15);
  v16 = sub_1E340A0(a6);
  sub_16BD430((__int64)v39, v16);
  sub_16BD4B0((__int64)v39, *(_QWORD *)(a6 + 24));
  v35[0] = 0;
  v17 = sub_1D17920((__int64)a1, (__int64)v39, a3, (__int64 *)v35);
  v22 = v17;
  if ( !v17 )
  {
    v24 = a1[26];
    v26 = *(_DWORD *)(a3 + 8);
    if ( v24 )
      a1[26] = *(_QWORD *)v24;
    else
      v24 = sub_145CBF0(a1 + 27, 112, 8);
    *((_QWORD *)&v29 + 1) = a10;
    *(_QWORD *)&v29 = a9;
    sub_1D189E0(v24, a2, v26, (unsigned __int8 **)a3, a4, a5, v29, a6);
    *(_BYTE *)(v24 + 26) |= 2u;
    sub_1D23B60((__int64)a1, v24, (__int64)a7, a8);
    sub_16BDA20(a1 + 40, (__int64 *)v24, v35[0]);
    if ( (_BYTE *)v39[0] != v40 )
      _libc_free(v39[0]);
LABEL_14:
    sub_1D172A0((__int64)a1, v24);
    return v24;
  }
  sub_1E34340(v17[13], a6, v18, v19, v20, v21);
  result = (__int64)v22;
  if ( (_BYTE *)v39[0] != v40 )
  {
    _libc_free(v39[0]);
    return (__int64)v22;
  }
  return result;
}
