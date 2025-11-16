// Function: sub_2585160
// Address: 0x2585160
//
__int64 __fastcall sub_2585160(__int64 a1, __int64 a2)
{
  __m128i *v2; // r13
  __int64 v3; // r12
  __int64 v4; // rax
  __int8 v5; // dl
  __int64 result; // rax
  __m128i v7; // xmm0
  bool v8; // zf
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r9
  char v13; // al
  __int64 v14; // r11
  _QWORD *v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // r13
  __int64 v18; // r14
  __int64 v19; // r8
  __int64 v20; // rbx
  __int64 v21; // r12
  __int64 v22; // rax
  char v23; // al
  __int64 v24; // r8
  unsigned __int64 v25; // r9
  __int64 *v26; // rax
  __int64 *i; // rdx
  unsigned __int64 v28; // r13
  int v29; // eax
  char v30; // r8
  __int64 v31; // [rsp+18h] [rbp-158h]
  __int64 v32; // [rsp+18h] [rbp-158h]
  __int8 v33; // [rsp+20h] [rbp-150h]
  unsigned __int64 v34; // [rsp+20h] [rbp-150h]
  unsigned int v35; // [rsp+28h] [rbp-148h]
  char v36; // [rsp+33h] [rbp-13Dh] BYREF
  int v37; // [rsp+34h] [rbp-13Ch] BYREF
  __int64 v38; // [rsp+38h] [rbp-138h] BYREF
  __m128i v39; // [rsp+40h] [rbp-130h] BYREF
  _QWORD v40[2]; // [rsp+50h] [rbp-120h] BYREF
  _QWORD v41[2]; // [rsp+60h] [rbp-110h] BYREF
  __m128i v42; // [rsp+70h] [rbp-100h] BYREF
  __int64 v43; // [rsp+80h] [rbp-F0h]
  __int64 *v44; // [rsp+90h] [rbp-E0h] BYREF
  __int64 v45; // [rsp+98h] [rbp-D8h]
  _QWORD v46[2]; // [rsp+A0h] [rbp-D0h] BYREF
  __m128i v47; // [rsp+B0h] [rbp-C0h] BYREF
  _QWORD v48[22]; // [rsp+C0h] [rbp-B0h] BYREF

  v2 = (__m128i *)(a1 + 72);
  v3 = a2;
  v45 = 0x100000000LL;
  LOBYTE(v40[0]) = 0;
  v44 = v46;
  v47.m128i_i32[0] = 81;
  sub_2515D00(a2, (__m128i *)(a1 + 72), v47.m128i_i32, 1, (__int64)&v44, 1);
  if ( (_DWORD)v45
    && (unsigned __int8)sub_2523890(
                          a2,
                          (__int64 (__fastcall *)(__int64, __int64 *))sub_253A490,
                          (__int64)&v47,
                          a1,
                          1u,
                          v40) )
  {
    v4 = sub_A72A60(v44);
    v5 = 1;
  }
  else
  {
    v42 = 0;
    LODWORD(v41[0]) = sub_250CB50(v2->m128i_i64, 0);
    v47.m128i_i64[0] = (__int64)v41;
    v47.m128i_i64[1] = a2;
    v48[0] = a1;
    v48[1] = &v42;
    if ( (unsigned __int8)sub_2523890(
                            a2,
                            (__int64 (__fastcall *)(__int64, __int64 *))sub_2587840,
                            (__int64)&v47,
                            a1,
                            1u,
                            v40) )
    {
      v4 = v42.m128i_i64[0];
      v5 = v42.m128i_i8[8];
      v39 = _mm_loadu_si128(&v42);
    }
    else
    {
      v5 = 1;
      v4 = 0;
    }
  }
  if ( v44 != v46 )
  {
    v31 = v4;
    v33 = v5;
    _libc_free((unsigned __int64)v44);
    v4 = v31;
    v5 = v33;
  }
  v39.m128i_i64[0] = v4;
  result = 1;
  v39.m128i_i8[8] = v5;
  v7 = _mm_loadu_si128(&v39);
  *(__m128i *)(a1 + 104) = v7;
  v8 = *(_BYTE *)(a1 + 112) == 0;
  v47 = v7;
  if ( !v8 )
  {
    if ( !*(_QWORD *)(a1 + 104) )
    {
      *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
      return 0;
    }
    v9 = sub_250D070(v2);
    v10 = sub_250D2C0(v9, 0);
    sub_2584D90(a2, v10, v11, a1, 1, 0, 1);
    v47.m128i_i32[0] = 81;
    if ( !(unsigned __int8)sub_2516400(a2, v2, (__int64)&v47, 1, 0, 0)
      && !(unsigned __int8)sub_254BF70(*(_QWORD *)(a1 + 104), *(_QWORD *)(*(_QWORD *)(a2 + 208) + 104LL)) )
    {
      v8 = *(_BYTE *)(a1 + 112) == 0;
      *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
      if ( v8 )
      {
        *(_QWORD *)(a1 + 104) = 0;
        *(_BYTE *)(a1 + 112) = 1;
        return 0;
      }
      else
      {
        *(_QWORD *)(a1 + 104) = 0;
        return 0;
      }
    }
    v12 = *(_QWORD *)(a1 + 104);
    v47.m128i_i64[0] = (__int64)v48;
    v47.m128i_i64[1] = 0x1000000000LL;
    v13 = *(_BYTE *)(v12 + 8);
    if ( v13 == 15 )
    {
      v14 = *(unsigned int *)(v12 + 12);
      if ( (_DWORD)v14 )
      {
        v15 = v48;
        v16 = 0;
        v17 = 8 * v14;
        v18 = v12;
        v19 = **(_QWORD **)(v12 + 16);
        v20 = 8;
        v21 = v19;
        while ( 1 )
        {
          v15[v16] = v21;
          v16 = (unsigned int)++v47.m128i_i32[2];
          if ( v20 == v17 )
            break;
          v21 = *(_QWORD *)(*(_QWORD *)(v18 + 16) + v20);
          if ( v16 + 1 > (unsigned __int64)v47.m128i_u32[3] )
          {
            sub_C8D5F0((__int64)&v47, v48, v16 + 1, 8u, v19, v12);
            v16 = v47.m128i_u32[2];
          }
          v15 = (_QWORD *)v47.m128i_i64[0];
          v20 += 8;
        }
        v2 = (__m128i *)(a1 + 72);
        v3 = a2;
      }
      goto LABEL_23;
    }
    if ( v13 != 16 )
    {
      v48[0] = v12;
      v47.m128i_i32[2] = 1;
LABEL_23:
      v22 = sub_25096F0(v2);
      v38 = sub_255ED30(*(_QWORD *)(*(_QWORD *)(v3 + 208) + 240LL), v22, 0);
      if ( !v38 )
        goto LABEL_25;
      v40[1] = &v47;
      v40[0] = &v38;
      v36 = 0;
      if ( !(unsigned __int8)sub_2523890(
                               v3,
                               (__int64 (__fastcall *)(__int64, __int64 *))sub_253AE30,
                               (__int64)v40,
                               a1,
                               1u,
                               &v36) )
        goto LABEL_25;
      v28 = sub_250C680(v2->m128i_i64);
      if ( !(unsigned __int8)sub_2523DA0(v3, v28) )
        goto LABEL_25;
      v29 = *(_DWORD *)(v28 + 32);
      v42.m128i_i64[1] = v3;
      v37 = v29;
      v42.m128i_i64[0] = (__int64)&v37;
      v44 = (__int64 *)&v37;
      v41[0] = &v42;
      v43 = a1;
      v45 = v3;
      v46[0] = a1;
      v41[1] = &v44;
      v30 = sub_2523890(v3, (__int64 (__fastcall *)(__int64, __int64 *))sub_2587AA0, (__int64)v41, a1, 1u, &v36);
      result = 1;
      if ( !v30 )
      {
LABEL_25:
        v23 = *(_BYTE *)(a1 + 96);
        v8 = *(_BYTE *)(a1 + 112) == 0;
        *(_QWORD *)(a1 + 104) = 0;
        *(_BYTE *)(a1 + 97) = v23;
        if ( v8 )
          *(_BYTE *)(a1 + 112) = 1;
        result = 0;
      }
      if ( (_QWORD *)v47.m128i_i64[0] != v48 )
      {
        v35 = result;
        _libc_free(v47.m128i_u64[0]);
        return v35;
      }
      return result;
    }
    v24 = *(_QWORD *)(v12 + 24);
    v25 = *(_QWORD *)(v12 + 32);
    if ( v25 > 0x10 )
    {
      v32 = v24;
      v34 = v25;
      sub_C8D5F0((__int64)&v47, v48, v25, 8u, v24, v25);
      v25 = v34;
      v24 = v32;
      v26 = (__int64 *)(v47.m128i_i64[0] + 8LL * v47.m128i_u32[2]);
    }
    else
    {
      if ( !v25 )
      {
LABEL_39:
        v47.m128i_i32[2] = v25;
        goto LABEL_23;
      }
      v26 = v48;
    }
    for ( i = &v26[v25]; i != v26; ++v26 )
      *v26 = v24;
    LODWORD(v25) = v47.m128i_i32[2] + v25;
    goto LABEL_39;
  }
  return result;
}
