// Function: sub_2665490
// Address: 0x2665490
//
__int64 __fastcall sub_2665490(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rsi
  __m128i *v3; // rdi
  unsigned __int64 v4; // xmm0_8
  __m128i v5; // xmm2
  __m128i v6; // xmm3
  unsigned __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __m128i *v11; // rax
  _BYTE *v12; // rcx
  __int64 v14; // [rsp+0h] [rbp-140h]
  unsigned __int64 v15; // [rsp+10h] [rbp-130h]
  unsigned __int64 v16; // [rsp+20h] [rbp-120h]
  _BYTE v17[16]; // [rsp+30h] [rbp-110h] BYREF
  void (__fastcall *v18)(_BYTE *, _BYTE *, __int64); // [rsp+40h] [rbp-100h]
  unsigned __int8 (__fastcall *v19)(_BYTE *); // [rsp+48h] [rbp-F8h]
  __m128i v20; // [rsp+50h] [rbp-F0h]
  __m128i v21; // [rsp+60h] [rbp-E0h]
  _BYTE v22[16]; // [rsp+70h] [rbp-D0h] BYREF
  void (__fastcall *v23)(_BYTE *, _BYTE *, __int64); // [rsp+80h] [rbp-C0h]
  __int64 v24; // [rsp+88h] [rbp-B8h]
  __m128i v25; // [rsp+90h] [rbp-B0h] BYREF
  __m128i v26; // [rsp+A0h] [rbp-A0h] BYREF
  _BYTE v27[16]; // [rsp+B0h] [rbp-90h] BYREF
  void (__fastcall *v28)(_BYTE *, _BYTE *, __int64); // [rsp+C0h] [rbp-80h]
  unsigned __int8 (__fastcall *v29)(_BYTE *); // [rsp+C8h] [rbp-78h]
  __m128i v30; // [rsp+D0h] [rbp-70h] BYREF
  __m128i v31; // [rsp+E0h] [rbp-60h] BYREF
  _BYTE v32[16]; // [rsp+F0h] [rbp-50h] BYREF
  void (__fastcall *v33)(_BYTE *, _BYTE *, __int64); // [rsp+100h] [rbp-40h]
  __int64 v34; // [rsp+108h] [rbp-38h]

  v1 = *(_QWORD *)(a1 + 80);
  v14 = a1 + 72;
  if ( v1 != a1 + 72 )
  {
    while ( 2 )
    {
      v2 = v1 - 24;
      v3 = &v25;
      if ( !v1 )
        v2 = 0;
      sub_AA69B0(&v25, v2, 1);
      v4 = _mm_loadu_si128(&v25).m128i_u64[0];
      v18 = 0;
      v15 = v4;
      v16 = _mm_loadu_si128(&v26).m128i_u64[0];
      if ( v28 )
      {
        v3 = (__m128i *)v17;
        v28(v17, v27, 2);
        v19 = v29;
        v18 = v28;
      }
      v5 = _mm_loadu_si128(&v30);
      v6 = _mm_loadu_si128(&v31);
      v23 = 0;
      v20 = v5;
      v21 = v6;
      if ( v33 )
      {
        v3 = (__m128i *)v22;
        v33(v22, v32, 2);
        v24 = v34;
        v23 = v33;
      }
LABEL_8:
      v7 = v15;
LABEL_9:
      while ( v20.m128i_i64[0] != v7 )
      {
        if ( !v7 )
          BUG();
        if ( *(_BYTE *)(v7 - 24) == 85 )
        {
          v9 = *(_QWORD *)(v7 - 56);
          if ( v9 )
          {
            if ( !*(_BYTE *)v9 && *(_QWORD *)(v9 + 24) == *(_QWORD *)(v7 + 56) && (*(_BYTE *)(v9 + 33) & 0x20) != 0 )
            {
              v10 = 2LL * (*(_DWORD *)(v7 - 20) & 0x7FFFFFF);
              if ( (*(_BYTE *)(v7 - 17) & 0x40) != 0 )
              {
                v11 = *(__m128i **)(v7 - 32);
                v3 = &v11[v10];
              }
              else
              {
                v3 = (__m128i *)(v7 - 24);
                v11 = (__m128i *)(v7 - 24 - v10 * 16);
              }
              if ( v11 != v3 )
              {
                while ( 1 )
                {
                  if ( *(_BYTE *)v11->m128i_i64[0] == 24 )
                  {
                    v12 = *(_BYTE **)(v11->m128i_i64[0] + 24);
                    if ( (unsigned __int8)(*v12 - 5) <= 0x1Fu && (v12[1] & 0x7F) == 1 )
                      break;
                  }
                  v11 += 2;
                  if ( v3 == v11 )
                    goto LABEL_12;
                }
                if ( v23 )
                  v23(v22, v22, 3);
                if ( v18 )
                  v18(v17, v17, 3);
                if ( v33 )
                  v33(v32, v32, 3);
                if ( v28 )
                  v28(v27, v27, 3);
                return 1;
              }
            }
          }
        }
LABEL_12:
        v7 = *(_QWORD *)(v7 + 8);
        v8 = 0;
        v15 = v7;
        if ( v7 != v16 )
        {
          while ( 1 )
          {
            if ( v7 )
              v7 -= 24LL;
            if ( !v18 )
              sub_4263D6(v3, v7, v8);
            v3 = (__m128i *)v17;
            if ( v19(v17) )
              goto LABEL_8;
            v7 = *(_QWORD *)(v15 + 8);
            v15 = v7;
            if ( v16 == v7 )
              goto LABEL_9;
          }
        }
      }
      if ( v23 )
        v23(v22, v22, 3);
      if ( v18 )
        v18(v17, v17, 3);
      if ( v33 )
        v33(v32, v32, 3);
      if ( v28 )
        v28(v27, v27, 3);
      v1 = *(_QWORD *)(v1 + 8);
      if ( v14 != v1 )
        continue;
      break;
    }
  }
  return 0;
}
