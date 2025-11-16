// Function: sub_38961C0
// Address: 0x38961c0
//
__int64 __fastcall sub_38961C0(__int64 a1, __int64 a2, int *a3, unsigned int a4)
{
  __int64 v7; // rdi
  __int64 v9; // r15
  __int64 v12; // rax
  unsigned __int64 v13; // rcx
  __int64 v14; // rdx
  int v15; // r15d
  __m128i *v16; // rsi
  __int64 v17; // [rsp+10h] [rbp-B0h]
  unsigned __int64 *v18; // [rsp+20h] [rbp-A0h]
  int v19; // [rsp+28h] [rbp-98h]
  int v20; // [rsp+30h] [rbp-90h] BYREF
  __int32 v21; // [rsp+34h] [rbp-8Ch] BYREF
  __int64 v22; // [rsp+38h] [rbp-88h] BYREF
  __m128i v23; // [rsp+40h] [rbp-80h] BYREF
  unsigned __int64 v24[4]; // [rsp+50h] [rbp-70h] BYREF
  __m128i v25; // [rsp+70h] [rbp-50h] BYREF
  __int64 v26; // [rsp+80h] [rbp-40h]
  __int64 v27; // [rsp+88h] [rbp-38h]

  v7 = a1 + 8;
  v9 = *(_QWORD *)(v7 + 48);
  *(_DWORD *)(a1 + 64) = sub_3887100(v7);
  v23 = 0u;
  LOBYTE(v20) = 0;
  if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here") )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here") )
    return 1;
  if ( (unsigned __int8)sub_388F6D0(a1, &v23) )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 4, "expected ',' here") )
    return 1;
  if ( (unsigned __int8)sub_388F470(a1, &v20) )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 4, "expected ',' here") )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 329, "expected 'aliasee' here") )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here") )
    return 1;
  v22 = 0;
  if ( (unsigned __int8)sub_388F790(a1, &v22, &v21) || (unsigned __int8)sub_388AF10(a1, 13, "expected ')' here") )
  {
    return 1;
  }
  else
  {
    v19 = v20;
    v12 = sub_22077B0(0x50u);
    v13 = v12;
    if ( v12 )
    {
      *(_DWORD *)(v12 + 8) = 0;
      *(_QWORD *)(v12 + 16) = 0;
      *(_DWORD *)(v12 + 12) = v19;
      *(_QWORD *)(v12 + 40) = 0;
      *(_QWORD *)(v12 + 48) = 0;
      *(_QWORD *)(v12 + 56) = 0;
      *(_QWORD *)(v12 + 64) = 0;
      *(_QWORD *)(v12 + 72) = 0;
      *(_QWORD *)v12 = &unk_49EB498;
    }
    v14 = qword_5052688;
    *(__m128i *)(v12 + 24) = v23;
    if ( (v22 & 0xFFFFFFFFFFFFFFF8LL) == (v14 & 0xFFFFFFFFFFFFFFF8LL) )
    {
      v17 = v12;
      memset(v24, 0, 24);
      v25.m128i_i32[0] = v21;
      v25.m128i_i64[1] = 0;
      v26 = 0;
      v27 = 0;
      v18 = (unsigned __int64 *)sub_38914E0((_QWORD *)(a1 + 1272), v25.m128i_i32);
      sub_388FA40(&v25.m128i_u64[1]);
      sub_388FA40(v24);
      v25.m128i_i64[1] = v9;
      v13 = v17;
      v25.m128i_i64[0] = v17;
      v16 = (__m128i *)v18[6];
      if ( v16 == (__m128i *)v18[7] )
      {
        sub_3894E60(v18 + 5, v16, &v25);
        v13 = v17;
      }
      else
      {
        if ( v16 )
          *v16 = _mm_loadu_si128(&v25);
        v18[6] += 16LL;
      }
    }
    else
    {
      *(_QWORD *)(v12 + 64) = **(_QWORD **)((v22 & 0xFFFFFFFFFFFFFFF8LL) + 24);
    }
    v24[0] = v13;
    v15 = v20 & 0xF;
    sub_2241BD0(v25.m128i_i64, a2);
    sub_3895460(a1, (__int64)&v25, a3, v15, a4, v24);
    sub_2240A30((unsigned __int64 *)&v25);
    sub_14EF120((__int64 *)v24);
    return 0;
  }
}
