// Function: sub_39D9B20
// Address: 0x39d9b20
//
__int64 **__fastcall sub_39D9B20(void **a1, __m128i *a2, __int64 a3, __int64 a4)
{
  __int64 **result; // rax
  __int64 *v5; // rbx
  __int64 v6; // rsi
  __int64 v7; // rcx
  __m128i *v8; // rdi
  __m128i si128; // xmm0
  __int64 v10; // rsi
  __int64 v11; // rdx
  __m128i *v12; // rdx
  __m128i v13; // xmm1
  __m128i *v14; // r14
  __m128i v15; // xmm5
  bool v16; // zf
  char *v17; // rbx
  __m128i *v18; // r13
  const __m128i *v19; // r12
  const __m128i *v20; // rbx
  const __m128i *v21; // r13
  __int64 **v22; // [rsp+0h] [rbp-160h]
  __int32 v23; // [rsp+14h] [rbp-14Ch]
  __int64 **v25; // [rsp+20h] [rbp-140h]
  __int64 *v26; // [rsp+38h] [rbp-128h]
  void **v27; // [rsp+40h] [rbp-120h] BYREF
  __int64 v28; // [rsp+48h] [rbp-118h]
  _QWORD v29[2]; // [rsp+50h] [rbp-110h] BYREF
  __m128i *v30; // [rsp+60h] [rbp-100h] BYREF
  __int64 v31; // [rsp+68h] [rbp-F8h]
  __m128i v32; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v33; // [rsp+80h] [rbp-E0h]
  __int64 v34; // [rsp+88h] [rbp-D8h]
  __m128i v35; // [rsp+90h] [rbp-D0h] BYREF
  __m128i v36; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v37; // [rsp+B0h] [rbp-B0h]
  const __m128i *v38; // [rsp+B8h] [rbp-A8h] BYREF
  __m128i *v39; // [rsp+C0h] [rbp-A0h]
  const __m128i *v40; // [rsp+C8h] [rbp-98h]
  void *v41; // [rsp+D0h] [rbp-90h] BYREF
  __int64 v42; // [rsp+D8h] [rbp-88h]
  __int64 v43; // [rsp+E0h] [rbp-80h]
  __int64 v44; // [rsp+E8h] [rbp-78h]
  int v45; // [rsp+F0h] [rbp-70h]
  void ***v46; // [rsp+F8h] [rbp-68h]
  __m128i v47; // [rsp+100h] [rbp-60h] BYREF
  __m128i v48; // [rsp+110h] [rbp-50h] BYREF
  __m128i v49[4]; // [rsp+120h] [rbp-40h] BYREF

  *(_DWORD *)a3 = *(_DWORD *)a4;
  result = *(__int64 ***)(a4 + 16);
  v22 = result;
  v25 = *(__int64 ***)(a4 + 8);
  v23 = 0;
  if ( v25 != result )
  {
    while ( 1 )
    {
      v28 = 0;
      LOBYTE(v29[0]) = 0;
      v27 = (void **)v29;
      v36.m128i_i32[0] = v23;
      v36.m128i_i64[1] = 0;
      v37 = 0;
      v38 = 0;
      v39 = 0;
      v40 = 0;
      v5 = *v25;
      ++v23;
      v26 = v25[1];
      if ( v26 != *v25 )
        break;
LABEL_27:
      v14 = *(__m128i **)(a3 + 16);
      if ( v14 == *(__m128i **)(a3 + 24) )
      {
        a2 = *(__m128i **)(a3 + 16);
        sub_39D94E0(a3 + 8, a2, &v36);
        v19 = v38;
      }
      else
      {
        if ( v14 )
        {
          v15 = _mm_load_si128(&v36);
          v14[2].m128i_i64[1] = 0;
          *v14 = v15;
          v14[1].m128i_i64[0] = v37;
          v17 = (char *)((char *)v39 - (char *)v38);
          v16 = v39 == v38;
          v14[2].m128i_i64[0] = 0;
          v14[1].m128i_i64[1] = 0;
          if ( v16 )
          {
            v18 = 0;
          }
          else
          {
            if ( (unsigned __int64)v17 > 0x7FFFFFFFFFFFFFE0LL )
              sub_4261EA(a1, a2, a3);
            v18 = (__m128i *)sub_22077B0((unsigned __int64)v17);
          }
          v14[1].m128i_i64[1] = (__int64)v18;
          v19 = v38;
          v14[2].m128i_i64[0] = (__int64)v18;
          v14[2].m128i_i64[1] = (__int64)&v17[(_QWORD)v18];
          v20 = v39;
          if ( v39 != v19 )
          {
            do
            {
              if ( v18 )
              {
                v18->m128i_i64[0] = (__int64)v18[1].m128i_i64;
                a2 = (__m128i *)v19->m128i_i64[0];
                sub_39CF630(v18->m128i_i64, v19->m128i_i64[0], v19->m128i_i64[0] + v19->m128i_i64[1]);
                v18[2] = _mm_loadu_si128(v19 + 2);
              }
              v19 += 3;
              v18 += 3;
            }
            while ( v20 != v19 );
            v19 = v38;
          }
          v14[2].m128i_i64[0] = (__int64)v18;
          v14 = *(__m128i **)(a3 + 16);
        }
        else
        {
          v19 = v38;
        }
        *(_QWORD *)(a3 + 16) = v14 + 3;
      }
      v21 = v39;
      if ( v19 != v39 )
      {
        do
        {
          if ( (const __m128i *)v19->m128i_i64[0] != &v19[1] )
          {
            a2 = (__m128i *)(v19[1].m128i_i64[0] + 1);
            j_j___libc_free_0(v19->m128i_i64[0]);
          }
          v19 += 3;
        }
        while ( v21 != v19 );
        v21 = v38;
      }
      if ( v21 )
      {
        a2 = (__m128i *)((char *)v40 - (char *)v21);
        j_j___libc_free_0((unsigned __int64)v21);
      }
      a1 = v27;
      if ( v27 != v29 )
      {
        a2 = (__m128i *)(v29[0] + 1LL);
        j_j___libc_free_0((unsigned __int64)v27);
      }
      v25 += 3;
      result = v25;
      if ( v22 == v25 )
        return result;
    }
    while ( 1 )
    {
      v10 = *v5;
      v45 = 1;
      v44 = 0;
      v41 = &unk_49EFBE0;
      v43 = 0;
      v42 = 0;
      v46 = &v27;
      sub_1DD5B60(&v47, v10);
      if ( !v48.m128i_i64[0] )
        sub_4263D6(&v47, v10, v11);
      ((void (__fastcall *)(__m128i *, void **))v48.m128i_i64[1])(&v47, &v41);
      if ( v48.m128i_i64[0] )
        ((void (__fastcall *)(__m128i *, __m128i *, __int64))v48.m128i_i64[0])(&v47, &v47, 3);
      if ( v44 != v42 )
        sub_16E7BA0((__int64 *)&v41);
      v30 = &v32;
      sub_39CF630((__int64 *)&v30, *v46, (__int64)v46[1] + (_QWORD)*v46);
      v12 = v30;
      if ( v30 == &v32 )
        break;
      v6 = v32.m128i_i64[0];
      v7 = v31;
      v33 = (__int64)v30;
      v30 = &v32;
      v35.m128i_i64[0] = v32.m128i_i64[0];
      v34 = v31;
      v31 = 0;
      v32.m128i_i8[0] = 0;
      v47.m128i_i64[0] = (__int64)&v48;
      if ( v12 == &v35 )
        goto LABEL_24;
      v47.m128i_i64[0] = (__int64)v12;
      v48.m128i_i64[0] = v6;
LABEL_6:
      v47.m128i_i64[1] = v7;
      a2 = v39;
      v49[0] = 0u;
      if ( v39 == v40 )
      {
        sub_39D9850((unsigned __int64 *)&v38, v39, &v47);
        v8 = (__m128i *)v47.m128i_i64[0];
      }
      else
      {
        v8 = (__m128i *)v47.m128i_i64[0];
        if ( v39 )
        {
          v39->m128i_i64[0] = (__int64)v39[1].m128i_i64;
          if ( (__m128i *)v47.m128i_i64[0] == &v48 )
          {
            a2[1] = _mm_load_si128(&v48);
          }
          else
          {
            a2->m128i_i64[0] = v47.m128i_i64[0];
            a2[1].m128i_i64[0] = v48.m128i_i64[0];
          }
          v8 = &v48;
          a2->m128i_i64[1] = v47.m128i_i64[1];
          si128 = _mm_load_si128(v49);
          v47.m128i_i64[0] = (__int64)&v48;
          v47.m128i_i64[1] = 0;
          v48.m128i_i8[0] = 0;
          a2[2] = si128;
          a2 = v39;
        }
        a2 += 3;
        v39 = a2;
      }
      if ( v8 != &v48 )
      {
        a2 = (__m128i *)(v48.m128i_i64[0] + 1);
        j_j___libc_free_0((unsigned __int64)v8);
      }
      if ( v30 != &v32 )
      {
        a2 = (__m128i *)(v32.m128i_i64[0] + 1);
        j_j___libc_free_0((unsigned __int64)v30);
      }
      a1 = &v41;
      ++v5;
      v28 = 0;
      *(_BYTE *)v27 = 0;
      sub_16E7BC0((__int64 *)&v41);
      if ( v26 == v5 )
        goto LABEL_27;
    }
    v13 = _mm_load_si128(&v32);
    v7 = v31;
    v47.m128i_i64[0] = (__int64)&v48;
    v31 = 0;
    v32.m128i_i8[0] = 0;
    v35 = v13;
LABEL_24:
    v48 = _mm_load_si128(&v35);
    goto LABEL_6;
  }
  return result;
}
