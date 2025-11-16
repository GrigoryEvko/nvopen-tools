// Function: sub_390CE40
// Address: 0x390ce40
//
__int64 __fastcall sub_390CE40(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // r12
  __int64 *v4; // rbx
  int v5; // r15d
  __int64 *v6; // r12
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // r8
  __int64 i; // rdi
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rsi
  int v15; // edx
  int v16; // ecx
  __int64 result; // rax
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rbx
  __int64 v21; // r13
  unsigned __int8 v22; // cl
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r12
  __int64 v26; // r14
  __int64 v27; // r15
  __int64 v28; // rax
  __int64 v29; // r13
  __int64 v30; // r15
  __int64 v31; // rdi
  __m128i v32; // xmm0
  __int64 v33; // rdx
  __int64 v34; // [rsp+0h] [rbp-C0h]
  __int64 v35; // [rsp+8h] [rbp-B8h]
  __int64 v36; // [rsp+10h] [rbp-B0h]
  _QWORD *v37; // [rsp+18h] [rbp-A8h]
  __int64 v38; // [rsp+30h] [rbp-90h]
  __int64 v39; // [rsp+38h] [rbp-88h]
  __m128i v40; // [rsp+40h] [rbp-80h] BYREF
  __int64 v41; // [rsp+50h] [rbp-70h]
  int v42; // [rsp+58h] [rbp-68h]
  __m128i v43; // [rsp+60h] [rbp-60h] BYREF
  __m128i v44; // [rsp+70h] [rbp-50h] BYREF
  __int64 v45; // [rsp+80h] [rbp-40h]
  int v46; // [rsp+88h] [rbp-38h]

  v2 = a2;
  v3 = a1;
  v4 = *(__int64 **)(a1 + 32);
  if ( v4 != *(__int64 **)(a1 + 40) )
  {
    v5 = 0;
    v6 = *(__int64 **)(a1 + 40);
    while ( 1 )
    {
      v7 = *v4;
      if ( (*(_QWORD *)(*v4 + 96) & 0xFFFFFFFFFFFFFFF8LL) == *v4 + 96 )
      {
        v8 = sub_22077B0(0xE0u);
        v9 = v8;
        if ( v8 )
        {
          sub_38CF760(v8, 1, 0, v7);
          *(_QWORD *)(v9 + 56) = 0;
          *(_WORD *)(v9 + 48) = 0;
          *(_QWORD *)(v9 + 64) = v9 + 80;
          *(_QWORD *)(v9 + 72) = 0x2000000000LL;
          *(_QWORD *)(v9 + 112) = v9 + 128;
          *(_QWORD *)(v9 + 120) = 0x400000000LL;
        }
      }
      ++v4;
      *(_DWORD *)(v7 + 28) = v5;
      if ( v6 == v4 )
        break;
      ++v5;
    }
    v3 = a1;
    v2 = a2;
  }
  v10 = *(unsigned int *)(v2 + 16);
  if ( !(_DWORD)v10 )
    goto LABEL_17;
  for ( i = 0; i != v10; ++i )
  {
    v12 = *(_QWORD *)(*(_QWORD *)(v2 + 8) + 8 * i);
    v13 = *(_QWORD *)(v12 + 104);
    *(_DWORD *)(v12 + 32) = i;
    v14 = v12 + 96;
    if ( v13 != v14 )
    {
      v15 = 0;
      do
      {
        v16 = v15++;
        *(_DWORD *)(v13 + 20) = v16;
        v13 = *(_QWORD *)(v13 + 8);
      }
      while ( v13 != v14 );
    }
  }
  while ( (unsigned __int8)sub_390CD10((__int64 *)v3, (__int64 **)v2) )
  {
    result = *(_QWORD *)v3;
    if ( *(_BYTE *)(*(_QWORD *)v3 + 1481LL) )
      return result;
LABEL_17:
    ;
  }
  sub_390CD90(v3, v2);
  (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(v3 + 24) + 24LL))(*(_QWORD *)(v3 + 24), v3, v2);
  result = *(_QWORD *)(v3 + 32);
  v34 = *(_QWORD *)(v3 + 40);
  if ( result == v34 )
    return result;
  v35 = *(_QWORD *)(v3 + 32);
  v18 = v3;
  v37 = (_QWORD *)v2;
  do
  {
    v36 = *(_QWORD *)v35 + 96LL;
    if ( v36 == *(_QWORD *)(*(_QWORD *)v35 + 104LL) )
      goto LABEL_34;
    v19 = v18;
    v20 = *(_QWORD *)(*(_QWORD *)v35 + 104LL);
    v21 = v19;
    do
    {
      v22 = *(_BYTE *)(v20 + 16);
      if ( v22 > 6u )
        goto LABEL_36;
      if ( ((1LL << v22) & 0x56) != 0 && v22 == 2 )
        goto LABEL_32;
      if ( ((1LL << v22) & 0x56) != 0 )
      {
        if ( v22 == 1 )
        {
          v23 = *(_QWORD *)(v20 + 112);
          v24 = *(unsigned int *)(v20 + 120);
          v25 = *(_QWORD *)(v20 + 64);
          v39 = *(_QWORD *)(v20 + 56);
          v26 = *(unsigned int *)(v20 + 72);
        }
        else if ( v22 == 4 )
        {
          v23 = *(_QWORD *)(v20 + 88);
          v24 = *(unsigned int *)(v20 + 96);
          v25 = *(_QWORD *)(v20 + 64);
          v39 = *(_QWORD *)(v20 + 56);
          v26 = *(unsigned int *)(v20 + 72);
        }
        else
        {
          v25 = *(_QWORD *)(v20 + 64);
          v26 = *(unsigned int *)(v20 + 72);
          v39 = 0;
          v23 = *(_QWORD *)(v20 + 88);
          v24 = *(unsigned int *)(v20 + 96);
        }
      }
      else
      {
LABEL_36:
        if ( v22 != 12 )
          goto LABEL_32;
        v25 = *(_QWORD *)(v20 + 64);
        v26 = *(unsigned int *)(v20 + 72);
        v39 = 0;
        v23 = *(_QWORD *)(v20 + 112);
        v24 = *(unsigned int *)(v20 + 120);
      }
      v38 = v23 + 24 * v24;
      if ( v23 != v38 )
      {
        v27 = v23;
        v28 = v21;
        v29 = v27;
        v30 = v28;
        do
        {
          v40 = 0u;
          v41 = 0;
          v42 = 0;
          sub_390C140(&v43, v30, v37, v20, v29);
          v31 = *(_QWORD *)(v30 + 8);
          v32 = _mm_loadu_si128(&v44);
          v41 = v45;
          v40 = v32;
          v42 = v46;
          v33 = v29;
          v29 += 24;
          (*(void (__fastcall **)(__int64, __int64, __int64, __m128i *, __int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)v31 + 64LL))(
            v31,
            v30,
            v33,
            &v40,
            v25,
            v26,
            v43.m128i_i64[1],
            v43.m128i_u8[0],
            v39);
        }
        while ( v38 != v29 );
        v21 = v30;
      }
LABEL_32:
      v20 = *(_QWORD *)(v20 + 8);
    }
    while ( v36 != v20 );
    v18 = v21;
LABEL_34:
    v35 += 8;
    result = v35;
  }
  while ( v34 != v35 );
  return result;
}
