// Function: sub_CE7350
// Address: 0xce7350
//
__int64 __fastcall sub_CE7350(__int64 a1, __int64 a2, const void *a3, size_t a4)
{
  __m128i v8; // rax
  int v9; // r15d
  unsigned __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rdi
  unsigned __int64 v13; // rbx
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rcx
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rsi
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rax
  __m128i v21; // xmm0
  __m128i v22; // xmm1
  const void *v23; // [rsp+0h] [rbp-120h]
  size_t v24; // [rsp+8h] [rbp-118h]
  int v25; // [rsp+1Ch] [rbp-104h]
  __int64 v26; // [rsp+28h] [rbp-F8h]
  __m128i v27; // [rsp+30h] [rbp-F0h] BYREF
  __m128i v28; // [rsp+40h] [rbp-E0h] BYREF
  __m128i v29; // [rsp+50h] [rbp-D0h] BYREF
  const char *v30; // [rsp+60h] [rbp-C0h] BYREF
  __m128i v31; // [rsp+70h] [rbp-B0h]
  __int16 v32; // [rsp+80h] [rbp-A0h]
  _QWORD v33[4]; // [rsp+90h] [rbp-90h] BYREF
  __int16 v34; // [rsp+B0h] [rbp-70h]
  _QWORD *v35; // [rsp+C0h] [rbp-60h] BYREF
  unsigned __int64 v36; // [rsp+C8h] [rbp-58h]
  const void *v37; // [rsp+D0h] [rbp-50h]
  size_t v38; // [rsp+D8h] [rbp-48h]
  __int16 v39; // [rsp+E0h] [rbp-40h]

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x300000000LL;
  v26 = sub_B2BE50(a2);
  if ( (unsigned __int8)sub_B2D620(a2, a3, a4) )
  {
    v35 = (_QWORD *)sub_B2D7E0(a2, a3, a4);
    v8.m128i_i64[0] = sub_A72240((__int64 *)&v35);
    v24 = a4;
    v27 = v8;
    v23 = (const void *)(a1 + 16);
    v9 = 3;
    do
    {
      if ( !v27.m128i_i64[1] )
        break;
      v10 = sub_C931B0(v27.m128i_i64, ",", 1u, 0);
      if ( v10 == -1 )
      {
        v22 = _mm_loadu_si128(&v27);
        v29 = 0u;
        v28 = v22;
      }
      else
      {
        v11 = v10 + 1;
        if ( v10 + 1 > v27.m128i_i64[1] )
        {
          v11 = v27.m128i_i64[1];
          v12 = 0;
        }
        else
        {
          v12 = v27.m128i_i64[1] - v11;
        }
        v28.m128i_i64[0] = v27.m128i_i64[0];
        if ( v10 > v27.m128i_i64[1] )
          v10 = v27.m128i_u64[1];
        v29.m128i_i64[1] = v12;
        v29.m128i_i64[0] = v11 + v27.m128i_i64[0];
        v28.m128i_i64[1] = v10;
      }
      v13 = 0;
      v14 = sub_C935B0(&v28, byte_3F15413, 6, 0);
      v15 = v28.m128i_u64[1];
      if ( v14 < v28.m128i_i64[1] )
      {
        v15 = v14;
        v13 = v28.m128i_i64[1] - v14;
      }
      v36 = v13;
      v35 = (_QWORD *)(v15 + v28.m128i_i64[0]);
      v16 = sub_C93740((__int64 *)&v35, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL) + 1;
      if ( v16 > v36 )
        v16 = v36;
      v17 = v36 - v13 + v16;
      if ( v17 > v36 )
        v17 = v36;
      if ( sub_C93C90((__int64)v35, v17, 0, (unsigned __int64 *)&v35) || v35 != (_QWORD *)(unsigned int)v35 )
      {
        v32 = 1283;
        v30 = "can't parse integer attribute ";
        v34 = 770;
        v31 = v28;
        v39 = 1282;
        v33[0] = &v30;
        v33[2] = " in ";
        v35 = v33;
        v37 = a3;
        v38 = v24;
        sub_B6ECE0(v26, (__int64)&v35);
      }
      else
      {
        v25 = (int)v35;
      }
      v20 = *(unsigned int *)(a1 + 8);
      if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        sub_C8D5F0(a1, v23, v20 + 1, 4u, v18, v19);
        v20 = *(unsigned int *)(a1 + 8);
      }
      *(_DWORD *)(*(_QWORD *)a1 + 4 * v20) = v25;
      v21 = _mm_loadu_si128(&v29);
      ++*(_DWORD *)(a1 + 8);
      v27 = v21;
      --v9;
    }
    while ( v9 );
  }
  return a1;
}
