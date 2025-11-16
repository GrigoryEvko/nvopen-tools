// Function: sub_252B440
// Address: 0x252b440
//
__int64 __fastcall sub_252B440(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  bool v5; // zf
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __m128i v10; // xmm4
  __m128i v11; // xmm5
  char v12; // al
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r8
  __m128i v16; // xmm0
  __m128i v17; // xmm2
  __m128i v18; // xmm1
  __int64 v19; // rax
  int v20; // eax
  __int64 v21; // rcx
  __int64 v22; // r8
  __m128i v23; // xmm0
  __m128i v24; // xmm2
  __m128i v25; // xmm1
  char v26; // [rsp+8h] [rbp-158h]
  unsigned __int8 v27; // [rsp+8h] [rbp-158h]
  __m128i v28; // [rsp+10h] [rbp-150h] BYREF
  __m128i v29; // [rsp+20h] [rbp-140h] BYREF
  __m128i v30; // [rsp+30h] [rbp-130h] BYREF
  __m128i v31; // [rsp+40h] [rbp-120h] BYREF
  __m128i v32; // [rsp+50h] [rbp-110h] BYREF
  __m128i v33; // [rsp+60h] [rbp-100h] BYREF
  __int64 v34; // [rsp+70h] [rbp-F0h]
  __m128i v35; // [rsp+80h] [rbp-E0h] BYREF
  __m128i v36; // [rsp+90h] [rbp-D0h]
  __m128i v37; // [rsp+A0h] [rbp-C0h]
  __int64 v38; // [rsp+B0h] [rbp-B0h]
  __int64 v39; // [rsp+C0h] [rbp-A0h] BYREF
  __int64 v40; // [rsp+C8h] [rbp-98h]
  __int64 v41; // [rsp+D0h] [rbp-90h]
  __int64 v42; // [rsp+D8h] [rbp-88h]
  unsigned __int8 **v43; // [rsp+E0h] [rbp-80h]
  __int64 v44; // [rsp+E8h] [rbp-78h]
  _BYTE v45[112]; // [rsp+F0h] [rbp-70h] BYREF

  if ( !(unsigned __int8)sub_B46970((unsigned __int8 *)a2) )
  {
    result = sub_B46420(a2);
    if ( !(_BYTE)result )
      return result;
  }
  v5 = *(_BYTE *)a2 == 85;
  v39 = 0;
  v43 = (unsigned __int8 **)v45;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v44 = 0x800000000LL;
  if ( v5
    && (v13 = *(_QWORD *)(a2 - 32)) != 0
    && !*(_BYTE *)v13
    && *(_QWORD *)(v13 + 24) == *(_QWORD *)(a2 + 80)
    && (*(_BYTE *)(v13 + 33) & 0x20) != 0
    && (unsigned int)(*(_DWORD *)(v13 + 36) - 238) <= 7
    && ((1LL << (*(_BYTE *)(v13 + 36) + 18)) & 0xAD) != 0 )
  {
    sub_D67210(&v28, a2);
    v16 = _mm_loadu_si128(&v28);
    v17 = _mm_loadu_si128(&v29);
    LOBYTE(v34) = 1;
    v18 = _mm_loadu_si128(&v30);
    v31 = v16;
    v38 = v34;
    v12 = 1;
    v32 = v17;
    v33 = v18;
    v35 = v16;
    v36 = v17;
    v37 = v18;
    if ( !v16.m128i_i64[0] )
      goto LABEL_8;
    sub_25193F0((__int64)&v39, v35.m128i_i64, v16.m128i_i64[0], v14, v15, (__int64)&v35);
    if ( *(_BYTE *)a2 == 85 )
    {
      v19 = *(_QWORD *)(a2 - 32);
      if ( v19 )
      {
        if ( !*(_BYTE *)v19 && *(_QWORD *)(v19 + 24) == *(_QWORD *)(a2 + 80) && (*(_BYTE *)(v19 + 33) & 0x20) != 0 )
        {
          v20 = *(_DWORD *)(v19 + 36);
          if ( v20 == 238 || (unsigned int)(v20 - 240) <= 1 )
          {
            sub_D671D0(&v28, a2);
            v23 = _mm_loadu_si128(&v28);
            v24 = _mm_loadu_si128(&v29);
            LOBYTE(v34) = 1;
            v25 = _mm_loadu_si128(&v30);
            v31 = v23;
            v38 = v34;
            v12 = 1;
            v32 = v24;
            v33 = v25;
            v35 = v23;
            v36 = v24;
            v37 = v25;
            if ( !v23.m128i_i64[0] )
              goto LABEL_8;
            sub_25193F0((__int64)&v39, v35.m128i_i64, v23.m128i_i64[0], v21, v22, (__int64)&v35);
          }
        }
      }
    }
  }
  else
  {
    sub_D66840(&v31, (_BYTE *)a2);
    v10 = _mm_loadu_si128(&v32);
    v11 = _mm_loadu_si128(&v33);
    v35 = _mm_loadu_si128(&v31);
    v38 = v34;
    v36 = v10;
    v37 = v11;
    if ( !(_BYTE)v34 || !v35.m128i_i64[0] )
    {
      v12 = 1;
      goto LABEL_8;
    }
    sub_25193F0((__int64)&v39, v35.m128i_i64, v6, v7, v8, v9);
  }
  v12 = sub_252B2C0(a1, v43, (unsigned int)v44, a3);
LABEL_8:
  if ( v43 != (unsigned __int8 **)v45 )
  {
    v26 = v12;
    _libc_free((unsigned __int64)v43);
    v12 = v26;
  }
  v27 = v12;
  sub_C7D6A0(v40, 8LL * (unsigned int)v42, 8);
  return v27;
}
