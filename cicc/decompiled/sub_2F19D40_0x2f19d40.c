// Function: sub_2F19D40
// Address: 0x2f19d40
//
__int64 **__fastcall sub_2F19D40(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 **result; // rax
  __int64 *v5; // r13
  __int64 v6; // rsi
  __int64 v7; // rcx
  __int64 v8; // r15
  __int64 v9; // rdx
  __m128i *v10; // rax
  __m128i v11; // xmm0
  __m128i *v12; // rsi
  unsigned __int64 *v13; // rbx
  unsigned __int64 *v14; // r12
  __int64 **v15; // [rsp+8h] [rbp-168h]
  __int32 v17; // [rsp+24h] [rbp-14Ch]
  __int64 **v18; // [rsp+28h] [rbp-148h]
  __int64 *v19; // [rsp+38h] [rbp-138h]
  _BYTE *v20; // [rsp+40h] [rbp-130h] BYREF
  __int64 v21; // [rsp+48h] [rbp-128h]
  _BYTE v22[16]; // [rsp+50h] [rbp-120h] BYREF
  __m128i *v23; // [rsp+60h] [rbp-110h] BYREF
  __int64 v24; // [rsp+68h] [rbp-108h]
  __m128i v25; // [rsp+70h] [rbp-100h] BYREF
  __int64 v26; // [rsp+80h] [rbp-F0h]
  __int64 v27; // [rsp+88h] [rbp-E8h]
  __m128i v28; // [rsp+90h] [rbp-E0h] BYREF
  __m128i v29; // [rsp+A0h] [rbp-D0h] BYREF
  __int64 v30; // [rsp+B0h] [rbp-C0h]
  unsigned __int64 *v31; // [rsp+B8h] [rbp-B8h] BYREF
  __int64 v32; // [rsp+C0h] [rbp-B0h]
  __int64 v33; // [rsp+C8h] [rbp-A8h]
  __m128i v34; // [rsp+D0h] [rbp-A0h] BYREF
  __m128i v35; // [rsp+E0h] [rbp-90h] BYREF
  __int64 v36; // [rsp+F0h] [rbp-80h]
  __int64 v37; // [rsp+F8h] [rbp-78h]
  _QWORD v38[14]; // [rsp+100h] [rbp-70h] BYREF

  v17 = 0;
  *(_DWORD *)a3 = *(_DWORD *)a4;
  result = *(__int64 ***)(a4 + 16);
  v15 = result;
  v18 = *(__int64 ***)(a4 + 8);
  if ( result != v18 )
  {
    while ( 1 )
    {
      v21 = 0;
      v22[0] = 0;
      v20 = v22;
      v29.m128i_i32[0] = v17;
      v29.m128i_i64[1] = 0;
      v30 = 0;
      v31 = 0;
      v32 = 0;
      v33 = 0;
      v5 = *v18;
      ++v17;
      v19 = v18[1];
      if ( v19 != *v18 )
      {
        while ( 1 )
        {
          v8 = *v5;
          v38[5] = 0x100000000LL;
          memset(&v38[1], 0, 32);
          v38[0] = &unk_49DD210;
          v38[6] = &v20;
          sub_CB5980((__int64)v38, 0, 0, 0);
          sub_2E31000(&v34, v8);
          if ( !v35.m128i_i64[0] )
            sub_4263D6(&v34, v8, v9);
          ((void (__fastcall *)(__m128i *, _QWORD *))v35.m128i_i64[1])(&v34, v38);
          if ( v35.m128i_i64[0] )
            ((void (__fastcall *)(__m128i *, __m128i *, __int64))v35.m128i_i64[0])(&v34, &v34, 3);
          v23 = &v25;
          sub_2F07250((__int64 *)&v23, v20, (__int64)&v20[v21]);
          v10 = v23;
          if ( v23 == &v25 )
            break;
          v6 = v25.m128i_i64[0];
          v7 = v24;
          v26 = (__int64)v23;
          v28.m128i_i64[0] = v25.m128i_i64[0];
          v27 = v24;
          v23 = &v25;
          v24 = 0;
          v25.m128i_i8[0] = 0;
          v34.m128i_i64[0] = (__int64)&v35;
          if ( v10 == &v28 )
            goto LABEL_16;
          v34.m128i_i64[0] = (__int64)v10;
          v35.m128i_i64[0] = v6;
LABEL_6:
          v34.m128i_i64[1] = v7;
          v36 = 0;
          v37 = 0;
          sub_2F147D0((unsigned __int64 *)&v31, &v34);
          if ( (__m128i *)v34.m128i_i64[0] != &v35 )
            j_j___libc_free_0(v34.m128i_u64[0]);
          if ( v23 != &v25 )
            j_j___libc_free_0((unsigned __int64)v23);
          ++v5;
          v21 = 0;
          *v20 = 0;
          v38[0] = &unk_49DD210;
          sub_CB5840((__int64)v38);
          if ( v19 == v5 )
            goto LABEL_17;
        }
        v11 = _mm_load_si128(&v25);
        v7 = v24;
        v25.m128i_i8[0] = 0;
        v24 = 0;
        v34.m128i_i64[0] = (__int64)&v35;
        v28 = v11;
LABEL_16:
        v35 = _mm_load_si128(&v28);
        goto LABEL_6;
      }
LABEL_17:
      v12 = *(__m128i **)(a3 + 16);
      if ( v12 == *(__m128i **)(a3 + 24) )
        break;
      if ( !v12 )
      {
        v13 = (unsigned __int64 *)v32;
        v14 = v31;
        *(_QWORD *)(a3 + 16) = 48;
LABEL_25:
        if ( v13 != v14 )
        {
          do
          {
            if ( (unsigned __int64 *)*v14 != v14 + 2 )
              j_j___libc_free_0(*v14);
            v14 += 6;
          }
          while ( v13 != v14 );
          v14 = v31;
        }
        if ( v14 )
          j_j___libc_free_0((unsigned __int64)v14);
        goto LABEL_20;
      }
      *v12 = _mm_load_si128(&v29);
      v12[1].m128i_i64[0] = v30;
      v12[1].m128i_i64[1] = (__int64)v31;
      v12[2].m128i_i64[0] = v32;
      v12[2].m128i_i64[1] = v33;
      *(_QWORD *)(a3 + 16) += 48LL;
LABEL_20:
      if ( v20 != v22 )
        j_j___libc_free_0((unsigned __int64)v20);
      v18 += 4;
      result = v18;
      if ( v15 == v18 )
        return result;
    }
    sub_2F19A70((unsigned __int64 *)(a3 + 8), v12, &v29);
    v13 = (unsigned __int64 *)v32;
    v14 = v31;
    goto LABEL_25;
  }
  return result;
}
