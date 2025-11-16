// Function: sub_20B7E10
// Address: 0x20b7e10
//
_BOOL8 __fastcall sub_20B7E10(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        const void **a6,
        __m128i a7,
        double a8,
        __m128i a9,
        __int64 *a10,
        int a11,
        __int128 a12,
        __int128 a13,
        __int128 a14,
        __int128 a15)
{
  unsigned int v15; // r10d
  const void **v16; // r11
  __int128 *v19; // rcx
  __int64 v20; // rsi
  const void ***v21; // rsi
  _BOOL4 v22; // r14d
  unsigned __int64 v23; // rdi
  __int128 *v27; // [rsp+10h] [rbp-80h]
  __int64 v29; // [rsp+20h] [rbp-70h] BYREF
  int v30; // [rsp+28h] [rbp-68h]
  _QWORD v31[2]; // [rsp+30h] [rbp-60h] BYREF
  _BYTE v32[80]; // [rsp+40h] [rbp-50h] BYREF

  v15 = a5;
  v16 = a6;
  v19 = *(__int128 **)(a2 + 32);
  v20 = *(_QWORD *)(a2 + 72);
  v31[0] = v32;
  v31[1] = 0x200000000LL;
  v29 = v20;
  if ( v20 )
  {
    v27 = v19;
    sub_1623A60((__int64)&v29, v20, 2);
    v15 = a5;
    v16 = a6;
    v19 = v27;
  }
  v21 = *(const void ****)(a2 + 40);
  v30 = *(_DWORD *)(a2 + 64);
  v22 = sub_20B5C20(
          a1,
          *(unsigned __int16 *)(a2 + 24),
          *(unsigned __int8 *)v21,
          v21[1],
          (__int64)&v29,
          (__int64)v31,
          a7,
          a8,
          a9,
          *v19,
          *(__int128 *)((char *)v19 + 40),
          v15,
          v16,
          a10,
          a11,
          a12,
          a13,
          a14,
          a15);
  if ( v29 )
    sub_161E7C0((__int64)&v29, v29);
  v23 = v31[0];
  if ( v22 )
  {
    *(_QWORD *)a3 = *(_QWORD *)v31[0];
    *(_DWORD *)(a3 + 8) = *(_DWORD *)(v23 + 8);
    *(_QWORD *)a4 = *(_QWORD *)(v23 + 16);
    *(_DWORD *)(a4 + 8) = *(_DWORD *)(v23 + 24);
  }
  if ( (_BYTE *)v23 != v32 )
    _libc_free(v23);
  return v22;
}
