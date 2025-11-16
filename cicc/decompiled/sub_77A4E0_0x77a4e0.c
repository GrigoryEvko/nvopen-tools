// Function: sub_77A4E0
// Address: 0x77a4e0
//
__int64 __fastcall sub_77A4E0(const __m128i *a1, __int64 a2, __m128i *a3)
{
  __int64 v5; // r12
  __int64 v6; // r13
  unsigned int v7; // ecx
  __int64 v8; // rsi
  int v9; // edx
  unsigned int v10; // eax
  int *v11; // rdi
  int v12; // r8d
  int v13; // eax
  char v14; // al
  __int64 result; // rax
  int *v16; // rdx
  int v17; // eax
  __int64 *v18; // r12
  __int32 v19; // edi
  __int32 v20; // esi
  __int64 v21; // rcx
  __int64 v22; // r12
  unsigned int v23; // edx
  _DWORD *i; // rax
  __int64 v25; // rcx
  __int32 v26; // edi
  __int64 v27; // rsi
  unsigned int v28; // eax
  __int64 v29; // rdx
  __int64 v30; // r13
  _DWORD v31[13]; // [rsp+Ch] [rbp-34h] BYREF

  v5 = *(_QWORD *)(a2 + 24);
  v31[0] = 1;
  v6 = *(_QWORD *)(a2 + 8);
  *a3 = _mm_loadu_si128(a1 + 1);
  a3[1] = _mm_loadu_si128(a1 + 2);
  a3[2].m128i_i64[0] = a1[3].m128i_i64[0];
  v7 = a1[4].m128i_u32[0];
  v8 = a1[3].m128i_i64[1];
  v9 = a1[8].m128i_i32[0] + 1;
  a1[8].m128i_i32[0] = v9;
  v10 = v7 & v9;
  a1[2].m128i_i32[2] = v9;
  v11 = (int *)(v8 + 4LL * (v7 & v9));
  v12 = *v11;
  *v11 = v9;
  if ( !v12 )
  {
    v13 = a1[4].m128i_i32[1] + 1;
    a1[4].m128i_i32[1] = v13;
    if ( v7 >= 2 * v13 )
      goto LABEL_3;
    goto LABEL_11;
  }
  do
  {
    v10 = v7 & (v10 + 1);
    v16 = (int *)(v8 + 4LL * v10);
  }
  while ( *v16 );
  *v16 = v12;
  v17 = a1[4].m128i_i32[1] + 1;
  a1[4].m128i_i32[1] = v17;
  if ( v7 < 2 * v17 )
LABEL_11:
    sub_7702C0((__int64)&a1[3].m128i_i64[1]);
LABEL_3:
  a1[3].m128i_i64[0] = 0;
  if ( !v5 )
    goto LABEL_6;
  v14 = *(_BYTE *)(v5 + 40);
  if ( v14 != 20 )
  {
    if ( v14 )
      sub_721090();
LABEL_6:
    result = v31[0];
    if ( v31[0] )
    {
      if ( !v6 )
        return result;
      sub_77A250((__int64)a1, *(_QWORD *)(v6 + 8), v31);
      result = v31[0];
      if ( v31[0] )
        return result;
    }
    goto LABEL_19;
  }
  v18 = *(__int64 **)(v5 + 72);
  if ( !v18 )
    goto LABEL_6;
  while ( 1 )
  {
    if ( *((_BYTE *)v18 + 8) == 7 )
    {
      sub_77A250((__int64)a1, v18[2], v31);
      if ( !v31[0] )
        break;
    }
    v18 = (__int64 *)*v18;
    if ( !v18 )
      goto LABEL_6;
  }
LABEL_19:
  v19 = a1[2].m128i_i32[2];
  v20 = a1[4].m128i_i32[0];
  v21 = a1[3].m128i_i64[1];
  v22 = a1[2].m128i_i64[0];
  v23 = v20 & v19;
  for ( i = (_DWORD *)(v21 + 4LL * (v20 & (unsigned int)v19)); v19 != *i; i = (_DWORD *)(v21 + 4LL * v23) )
    v23 = v20 & (v23 + 1);
  *i = 0;
  if ( *(_DWORD *)(v21 + 4LL * ((v23 + 1) & v20)) )
    sub_771390(a1[3].m128i_i64[1], a1[4].m128i_i32[0], v23);
  --a1[4].m128i_i32[1];
  a1[1] = _mm_loadu_si128(a3);
  a1[2] = _mm_loadu_si128(a3 + 1);
  a1[3].m128i_i64[0] = a3[2].m128i_i64[0];
  if ( v22 && v22 != a3[1].m128i_i64[0] )
  {
    while ( 1 )
    {
      v25 = *(unsigned int *)(v22 + 12);
      v26 = a1[4].m128i_i32[0];
      v27 = a1[3].m128i_i64[1];
      v28 = v26 & *(_DWORD *)(v22 + 12);
      v29 = *(unsigned int *)(v27 + 4LL * v28);
      if ( (_DWORD)v25 == (_DWORD)v29 || !(_DWORD)v25 )
        break;
      while ( (_DWORD)v29 )
      {
        v28 = v26 & (v28 + 1);
        v29 = *(unsigned int *)(v27 + 4LL * v28);
        if ( (_DWORD)v25 == (_DWORD)v29 )
          goto LABEL_33;
      }
      v30 = *(_QWORD *)v22;
      sub_822B90(v22, *(unsigned int *)(v22 + 8), v29, v25);
      if ( !v30 )
      {
        v22 = 0;
        break;
      }
      v22 = v30;
    }
LABEL_33:
    a1[2].m128i_i64[0] = v22;
  }
  return v31[0];
}
