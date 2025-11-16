// Function: sub_E6DEB0
// Address: 0xe6deb0
//
unsigned __int64 __fastcall sub_E6DEB0(
        _QWORD *a1,
        _BYTE *a2,
        size_t a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        unsigned int a7,
        unsigned int a8)
{
  __int64 v8; // r14
  __m128i v9; // xmm0
  __m128i *v10; // rax
  __int64 *v11; // r10
  char v12; // dl
  char v13; // r13
  unsigned __int64 v14; // r13
  char v16; // al
  __int64 *v17; // r8
  __int64 v18; // rdx
  unsigned __int64 *v19; // rax
  int v20; // r11d
  unsigned __int64 *v21; // rbx
  __int64 v22; // rax
  __int64 *v23; // r10
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 *v26; // [rsp+8h] [rbp-D8h]
  __int64 v28; // [rsp+18h] [rbp-C8h]
  __int64 v29; // [rsp+18h] [rbp-C8h]
  __int64 v30; // [rsp+18h] [rbp-C8h]
  __int64 v31; // [rsp+18h] [rbp-C8h]
  __int64 v32; // [rsp+20h] [rbp-C0h]
  __m128i *v33; // [rsp+20h] [rbp-C0h]
  __int64 v34; // [rsp+20h] [rbp-C0h]
  __int64 v35; // [rsp+20h] [rbp-C0h]
  __int64 v36; // [rsp+20h] [rbp-C0h]
  __int64 *v38; // [rsp+28h] [rbp-B8h]
  _BYTE *v39[2]; // [rsp+30h] [rbp-B0h] BYREF
  _QWORD v40[2]; // [rsp+40h] [rbp-A0h] BYREF
  __m128i v41; // [rsp+50h] [rbp-90h] BYREF
  unsigned __int64 v42; // [rsp+60h] [rbp-80h]
  __m128i v43; // [rsp+70h] [rbp-70h] BYREF
  _QWORD v44[2]; // [rsp+80h] [rbp-60h] BYREF
  __m128i v45; // [rsp+90h] [rbp-50h]
  unsigned __int64 v46; // [rsp+A0h] [rbp-40h]
  __int64 v47; // [rsp+A8h] [rbp-38h]

  v8 = 0;
  if ( a6 )
  {
    v43.m128i_i64[0] = a5;
    v45.m128i_i16[0] = 261;
    v43.m128i_i64[1] = a6;
    v8 = sub_E6C460((__int64)a1, (const char **)&v43);
    v16 = *(_BYTE *)(v8 + 8);
    if ( (v16 & 1) != 0 )
    {
      v17 = *(__int64 **)(v8 - 8);
      a6 = *v17;
      a5 = (__int64)(v17 + 3);
    }
    else
    {
      a6 = 0;
      a5 = 0;
    }
    if ( a7 != 5 )
    {
      v18 = *(_QWORD *)v8;
      if ( *(_QWORD *)v8 )
        goto LABEL_14;
      if ( (*(_BYTE *)(v8 + 9) & 0x70) == 0x20 && v16 >= 0 )
      {
        *(_BYTE *)(v8 + 8) |= 8u;
        v31 = a6;
        v36 = a5;
        v24 = sub_E807D0(*(_QWORD *)(v8 + 24));
        a5 = v36;
        a6 = v31;
        *(_QWORD *)v8 = v24;
        v18 = v24;
        if ( v24 )
        {
LABEL_14:
          if ( (_UNKNOWN *)v18 == off_4C5D170 || *(_QWORD *)(*(_QWORD *)(v18 + 8) + 160LL) != v8 )
          {
            v29 = a6;
            v34 = a5;
            v43.m128i_i64[0] = (__int64)"invalid symbol redefinition";
            v45.m128i_i16[0] = 259;
            sub_E66880((__int64)a1, 0, (__int64)&v43);
            a6 = v29;
            a5 = v34;
          }
        }
      }
    }
  }
  v28 = a6;
  v32 = a5;
  v39[0] = v40;
  sub_E62BB0((__int64 *)v39, a2, (__int64)&a2[a3]);
  v43.m128i_i64[0] = (__int64)v44;
  v42 = __PAIR64__(a8, a7);
  v41.m128i_i64[0] = v32;
  v41.m128i_i64[1] = v28;
  sub_E62C60(v43.m128i_i64, v39[0], (__int64)&v39[0][(unsigned __int64)v39[1]]);
  v9 = _mm_loadu_si128(&v41);
  v47 = 0;
  v46 = v42;
  v45 = v9;
  v10 = sub_E6A4F0((__int64)(a1 + 250), &v43);
  v11 = (__int64 *)v10;
  v13 = v12;
  if ( (_QWORD *)v43.m128i_i64[0] != v44 )
  {
    v33 = v10;
    j_j___libc_free_0(v43.m128i_i64[0], v44[0] + 1LL);
    v11 = (__int64 *)v33;
  }
  if ( v13 )
  {
    v26 = v11;
    v30 = v11[5];
    v35 = v11[4];
    v19 = sub_E6CFF0((__int64)a1, a2, a3);
    v20 = v30;
    a1[58] += 176LL;
    v21 = v19;
    v22 = a1[48];
    v23 = v26;
    v14 = (v22 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( a1[49] >= v14 + 176 && v22 )
    {
      a1[48] = v14 + 176;
    }
    else
    {
      v25 = sub_9D1E70((__int64)(a1 + 48), 176, 176, 3);
      v23 = v26;
      v20 = v30;
      v14 = v25;
    }
    v38 = v23;
    sub_E92760(v14, 0, v35, v20, (a4 >> 5) & 1, (a4 >> 7) & 1, (__int64)v21);
    *(_QWORD *)(v14 + 160) = v8;
    *(_DWORD *)(v14 + 152) = -1;
    *(_QWORD *)v14 = &unk_49E35B0;
    *(_DWORD *)(v14 + 148) = a4;
    *(_DWORD *)(v14 + 168) = a7;
    v38[11] = v14;
    *v21 = sub_E6B260(a1, v14);
  }
  else
  {
    v14 = v11[11];
  }
  if ( (_QWORD *)v39[0] != v40 )
    j_j___libc_free_0(v39[0], v40[0] + 1LL);
  return v14;
}
