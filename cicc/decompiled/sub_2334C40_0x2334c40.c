// Function: sub_2334C40
// Address: 0x2334c40
//
__int64 __fastcall sub_2334C40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __m128i v4; // xmm1
  int v5; // edx
  size_t v6; // r12
  const void *v7; // rdi
  char v8; // cl
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // r15
  char v13; // [rsp+5h] [rbp-FBh]
  char v14; // [rsp+6h] [rbp-FAh]
  char v15; // [rsp+7h] [rbp-F9h]
  __int32 v16; // [rsp+8h] [rbp-F8h]
  char v17; // [rsp+Ch] [rbp-F4h]
  char v18; // [rsp+Dh] [rbp-F3h]
  char v19; // [rsp+Eh] [rbp-F2h]
  char v20; // [rsp+Fh] [rbp-F1h]
  const void *v21; // [rsp+10h] [rbp-F0h]
  char v22; // [rsp+18h] [rbp-E8h]
  char v23; // [rsp+19h] [rbp-E7h]
  char v24; // [rsp+1Ah] [rbp-E6h]
  char v25; // [rsp+1Bh] [rbp-E5h]
  int v26; // [rsp+1Ch] [rbp-E4h]
  char v28; // [rsp+28h] [rbp-D8h]
  unsigned int v29; // [rsp+28h] [rbp-D8h]
  __m128i v30; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v31; // [rsp+48h] [rbp-B8h] BYREF
  __int64 v32; // [rsp+54h] [rbp-ACh]
  int v33; // [rsp+5Ch] [rbp-A4h]
  __m128i v34; // [rsp+60h] [rbp-A0h] BYREF
  unsigned __int64 v35[4]; // [rsp+70h] [rbp-90h] BYREF
  __m128i v36; // [rsp+90h] [rbp-70h] BYREF
  __m128i v37; // [rsp+A0h] [rbp-60h] BYREF
  char v38; // [rsp+B0h] [rbp-50h]
  _QWORD v39[2]; // [rsp+B8h] [rbp-48h] BYREF
  _QWORD *v40; // [rsp+C8h] [rbp-38h] BYREF

  v3 = a1;
  v30.m128i_i64[0] = a2;
  v30.m128i_i64[1] = a3;
  v22 = 0;
  v19 = 0;
  v25 = 0;
  v26 = 2;
  v23 = 0;
  v20 = 0;
  v24 = 0;
  if ( a3 )
  {
    while ( 1 )
    {
      v34 = 0u;
      LOBYTE(v35[0]) = 59;
      sub_232E160(&v36, &v30, v35, 1u);
      v4 = _mm_loadu_si128(&v37);
      v34 = _mm_loadu_si128(&v36);
      v30 = v4;
      v32 = sub_232E370((const void *)v36.m128i_i64[0], v34.m128i_u64[1]);
      v33 = v5;
      if ( (_BYTE)v5 && !HIDWORD(v32) )
      {
        v26 = v32;
      }
      else if ( (unsigned __int8)sub_95CB50((const void **)&v34, "full-unroll-max=", 0x10u) )
      {
        if ( sub_C93CC0(v34.m128i_i64[0], v34.m128i_i64[1], 0, v36.m128i_i64) || v36.m128i_i64[0] != v36.m128i_i32[0] )
        {
LABEL_18:
          v3 = a1;
          v9 = sub_C63BB0();
          v38 = 1;
          v11 = v10;
          v39[1] = &v34;
          v36.m128i_i64[0] = (__int64)"invalid LoopUnrollPass parameter '{0}' ";
          v37.m128i_i64[0] = (__int64)&v40;
          v29 = v9;
          v36.m128i_i64[1] = 39;
          v39[0] = &unk_49DB108;
          v40 = v39;
          v37.m128i_i64[1] = 1;
          sub_23328D0((__int64)v35, (__int64)&v36);
          sub_23058C0(&v31, (__int64)v35, v29, v11);
          *(_BYTE *)(a1 + 32) |= 3u;
          *(_QWORD *)a1 = v31 & 0xFFFFFFFFFFFFFFFELL;
          sub_2240A30(v35);
          return v3;
        }
        v16 = v36.m128i_i32[0];
        v25 = 1;
      }
      else
      {
        v28 = sub_95CB50((const void **)&v34, "no-", 3u) ^ 1;
        if ( v34.m128i_i64[1] == 7
          && *(_DWORD *)v34.m128i_i64[0] == 1953653104
          && *(_WORD *)(v34.m128i_i64[0] + 4) == 24937
          && *(_BYTE *)(v34.m128i_i64[0] + 6) == 108 )
        {
          v19 = 1;
          v13 = v28;
        }
        else
        {
          v6 = v34.m128i_u64[1];
          v21 = (const void *)v34.m128i_i64[0];
          v7 = (const void *)v34.m128i_i64[0];
          if ( sub_9691B0((const void *)v34.m128i_i64[0], v34.m128i_u64[1], "peeling", 7) )
          {
            v24 = 1;
            v18 = v28;
          }
          else if ( sub_9691B0(v7, v6, "profile-peeling", 15) )
          {
            v8 = v23;
            v17 = v28;
            if ( !v23 )
              v8 = 1;
            v23 = v8;
          }
          else if ( sub_9691B0(v21, v6, "runtime", 7) )
          {
            v22 = 1;
            v15 = v28;
          }
          else
          {
            if ( !sub_9691B0(v21, v6, "upperbound", 10) )
              goto LABEL_18;
            v20 = 1;
            v14 = v28;
          }
        }
      }
      if ( !v30.m128i_i64[1] )
      {
        v3 = a1;
        break;
      }
    }
  }
  *(_BYTE *)(v3 + 32) = *(_BYTE *)(v3 + 32) & 0xFC | 2;
  *(_BYTE *)v3 = v13;
  *(_BYTE *)(v3 + 1) = v19;
  *(_BYTE *)(v3 + 2) = v18;
  *(_BYTE *)(v3 + 3) = v24;
  *(_BYTE *)(v3 + 4) = v15;
  *(_BYTE *)(v3 + 5) = v22;
  *(_BYTE *)(v3 + 6) = v14;
  *(_BYTE *)(v3 + 7) = v20;
  *(_BYTE *)(v3 + 8) = v17;
  *(_BYTE *)(v3 + 9) = v23;
  *(_DWORD *)(v3 + 12) = v16;
  *(_BYTE *)(v3 + 16) = v25;
  *(_DWORD *)(v3 + 20) = v26;
  *(_WORD *)(v3 + 24) = 0;
  return v3;
}
