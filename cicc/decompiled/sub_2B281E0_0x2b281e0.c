// Function: sub_2B281E0
// Address: 0x2b281e0
//
_QWORD *__fastcall sub_2B281E0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  _QWORD *v6; // r9
  int v7; // r10d
  _BYTE *v8; // r11
  bool v9; // zf
  signed __int64 v10; // rax
  unsigned int v11; // r12d
  unsigned __int64 v12; // rdx
  _BYTE **v13; // rdi
  unsigned __int64 v14; // xmm0_8
  unsigned __int64 v15; // rbx
  unsigned __int64 *v16; // rax
  unsigned int v17; // edx
  __int64 v18; // rdx
  __int64 v20; // rbx
  int v21; // [rsp+4h] [rbp-ECh]
  unsigned __int64 v22; // [rsp+8h] [rbp-E8h]
  _BYTE *v23; // [rsp+10h] [rbp-E0h]
  _BYTE **v24; // [rsp+18h] [rbp-D8h]
  _QWORD *v25; // [rsp+20h] [rbp-D0h]
  __int64 v26; // [rsp+28h] [rbp-C8h]
  _BYTE *v27; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v28; // [rsp+38h] [rbp-B8h]
  _BYTE v29[176]; // [rsp+40h] [rbp-B0h] BYREF

  v6 = a1;
  v7 = a4;
  v8 = v29;
  v26 = a3;
  v9 = *(_DWORD *)(a2 + 12) == 1;
  v27 = v29;
  v28 = 0x800000000LL;
  if ( v9 )
  {
    v10 = 0;
    v17 = 0;
  }
  else
  {
    v25 = a1;
    v10 = 0;
    v11 = 0;
    v12 = 8;
    v13 = &v27;
    a5 = 0xFFFFFFFF00000000LL;
    while ( 1 )
    {
      v14 = _mm_cvtsi32_si128(v11).m128i_u64[0];
      v15 = v5 & 0xFFFFFF0000000000LL;
      v5 &= 0xFFFFFF0000000000LL;
      if ( v10 + 1 > v12 )
      {
        v21 = v7;
        v23 = v8;
        v24 = v13;
        v22 = v14;
        sub_C8D5F0((__int64)v13, v8, v10 + 1, 0x10u, 0xFFFFFFFF00000000LL, v10 + 1);
        v10 = (unsigned int)v28;
        v7 = v21;
        a5 = 0xFFFFFFFF00000000LL;
        v14 = v22;
        v8 = v23;
        v13 = v24;
      }
      v16 = (unsigned __int64 *)&v27[16 * v10];
      ++v11;
      v16[1] = v15;
      *v16 = v14;
      a4 = *(unsigned int *)(a2 + 12);
      v10 = (unsigned int)(v28 + 1);
      v17 = a4 - 1;
      LODWORD(v28) = v28 + 1;
      if ( v11 >= (int)a4 - 1 )
        break;
      v12 = HIDWORD(v28);
    }
    v6 = v25;
  }
  if ( (_BYTE)v7 )
  {
    a4 = HIDWORD(v28);
    v20 = v17 | 0xA00000000LL;
    if ( v10 + 1 > (unsigned __int64)HIDWORD(v28) )
    {
      v24 = (_BYTE **)v6;
      v25 = v8;
      sub_C8D5F0((__int64)&v27, v8, v10 + 1, 0x10u, a5, (__int64)v6);
      v10 = (unsigned int)v28;
      v6 = v24;
      v8 = v25;
    }
    v10 = (signed __int64)&v27[16 * v10];
    *(_QWORD *)v10 = v20;
    *(_QWORD *)(v10 + 8) = 0;
    LODWORD(v10) = v28 + 1;
    LODWORD(v28) = v28 + 1;
  }
  v18 = v26;
  v6[2] = 0x800000000LL;
  *v6 = v18;
  v6[1] = v6 + 3;
  if ( (_DWORD)v10 )
  {
    v24 = (_BYTE **)v8;
    v25 = v6;
    sub_2B0D350((__int64)(v6 + 1), (__int64)&v27, (__int64)(v6 + 3), a4, a5, (__int64)v6);
    v8 = v24;
    v6 = v25;
  }
  if ( v27 != v8 )
  {
    v25 = v6;
    _libc_free((unsigned __int64)v27);
    return v25;
  }
  return v6;
}
