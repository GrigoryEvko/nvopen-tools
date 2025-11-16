// Function: sub_3474980
// Address: 0x3474980
//
__int64 __fastcall sub_3474980(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        __int64 a6,
        __m128i a7,
        __int64 a8,
        int a9,
        __int128 a10,
        __int128 a11,
        __int128 a12,
        __int128 a13)
{
  unsigned int v13; // r10d
  __int64 v14; // r11
  __int128 *v17; // rcx
  __int64 v18; // rsi
  unsigned __int16 *v19; // rsi
  unsigned int v20; // r14d
  unsigned __int64 v21; // rdi
  __int128 *v25; // [rsp+10h] [rbp-80h]
  __int64 v27; // [rsp+20h] [rbp-70h] BYREF
  int v28; // [rsp+28h] [rbp-68h]
  _QWORD v29[2]; // [rsp+30h] [rbp-60h] BYREF
  _BYTE v30[80]; // [rsp+40h] [rbp-50h] BYREF

  v13 = a5;
  v14 = a6;
  v17 = *(__int128 **)(a2 + 40);
  v18 = *(_QWORD *)(a2 + 80);
  v29[0] = v30;
  v29[1] = 0x200000000LL;
  v27 = v18;
  if ( v18 )
  {
    v25 = v17;
    sub_B96E90((__int64)&v27, v18, 1);
    v13 = a5;
    v14 = a6;
    v17 = v25;
  }
  v19 = *(unsigned __int16 **)(a2 + 48);
  v28 = *(_DWORD *)(a2 + 72);
  v20 = sub_3472970(
          a1,
          *(unsigned int *)(a2 + 24),
          *v19,
          *((_QWORD *)v19 + 1),
          (__int64)&v27,
          (__int64)v29,
          a7,
          *v17,
          *(__int128 *)((char *)v17 + 40),
          v13,
          v14,
          a8,
          a9,
          a10,
          a11,
          a12,
          a13);
  if ( v27 )
    sub_B91220((__int64)&v27, v27);
  v21 = v29[0];
  if ( (_BYTE)v20 )
  {
    *(_QWORD *)a3 = *(_QWORD *)v29[0];
    *(_DWORD *)(a3 + 8) = *(_DWORD *)(v21 + 8);
    *(_QWORD *)a4 = *(_QWORD *)(v21 + 16);
    *(_DWORD *)(a4 + 8) = *(_DWORD *)(v21 + 24);
  }
  if ( (_BYTE *)v21 != v30 )
    _libc_free(v21);
  return v20;
}
