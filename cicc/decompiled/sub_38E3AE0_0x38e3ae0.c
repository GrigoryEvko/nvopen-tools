// Function: sub_38E3AE0
// Address: 0x38e3ae0
//
_QWORD *__fastcall sub_38E3AE0(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        unsigned __int64 a5,
        int a6)
{
  __int64 v7; // r10
  __int64 v8; // r13
  unsigned __int64 v9; // rax
  __m128i v10; // xmm0
  __int64 *v11; // rdi
  unsigned __int64 v12; // rax
  unsigned int v13; // r8d
  size_t v14; // rdx
  _BYTE *v15; // rdi
  __int64 v16; // r13
  __int64 v17; // rbx
  unsigned __int64 v18; // rdi
  __int64 *v19; // rdi
  unsigned int v21; // [rsp+Ch] [rbp-114h]
  unsigned int v22; // [rsp+Ch] [rbp-114h]
  __int64 v26; // [rsp+40h] [rbp-E0h]
  unsigned __int64 v27[2]; // [rsp+50h] [rbp-D0h] BYREF
  _BYTE **v28; // [rsp+60h] [rbp-C0h] BYREF
  __int16 v29; // [rsp+70h] [rbp-B0h]
  unsigned __int64 v30; // [rsp+80h] [rbp-A0h] BYREF
  _BYTE *v31; // [rsp+88h] [rbp-98h] BYREF
  __int64 v32; // [rsp+90h] [rbp-90h]
  _BYTE dest[64]; // [rsp+98h] [rbp-88h] BYREF
  __m128i v34; // [rsp+D8h] [rbp-48h]

  v7 = *(_QWORD *)(a1 + 24);
  v26 = v7 + 104LL * *(unsigned int *)(a1 + 32);
  if ( v7 != v26 )
  {
    v8 = *(_QWORD *)(a1 + 24);
    do
    {
      v12 = *(_QWORD *)v8;
      v31 = dest;
      v30 = v12;
      v32 = 0x4000000000LL;
      v13 = *(_DWORD *)(v8 + 16);
      if ( v13 && &v31 != (_BYTE **)(v8 + 8) )
      {
        v14 = v13;
        v15 = dest;
        if ( v13 <= 0x40
          || (v22 = *(_DWORD *)(v8 + 16),
              sub_16CD150((__int64)&v31, dest, v13, 1, v13, a6),
              v14 = *(unsigned int *)(v8 + 16),
              v15 = v31,
              v13 = v22,
              *(_DWORD *)(v8 + 16)) )
        {
          v21 = v13;
          memcpy(v15, *(const void **)(v8 + 8), v14);
          v13 = v21;
        }
        LODWORD(v32) = v13;
      }
      v9 = *(_QWORD *)(v8 + 88);
      v10 = _mm_loadu_si128((const __m128i *)(v8 + 88));
      *(_BYTE *)(a1 + 17) = 1;
      v11 = *(__int64 **)(a1 + 344);
      v34 = v10;
      v28 = &v31;
      v27[0] = v9;
      v29 = 262;
      v27[1] = v10.m128i_u64[1];
      sub_16D14E0(v11, v30, 0, (__int64)&v28, v27, 1, 0, 0, 1u);
      sub_38E35B0((_QWORD *)a1);
      if ( v31 != dest )
        _libc_free((unsigned __int64)v31);
      v8 += 104;
    }
    while ( v26 != v8 );
    v16 = *(_QWORD *)(a1 + 24);
    v17 = v16 + 104LL * *(unsigned int *)(a1 + 32);
    while ( v16 != v17 )
    {
      v17 -= 104;
      v18 = *(_QWORD *)(v17 + 8);
      if ( v18 != v17 + 24 )
        _libc_free(v18);
    }
  }
  *(_DWORD *)(a1 + 32) = 0;
  v19 = *(__int64 **)(a1 + 344);
  v30 = a4;
  v31 = (_BYTE *)a5;
  sub_16D14E0(v19, a2, 3, a3, &v30, 1, 0, 0, 1u);
  return sub_38E35B0((_QWORD *)a1);
}
