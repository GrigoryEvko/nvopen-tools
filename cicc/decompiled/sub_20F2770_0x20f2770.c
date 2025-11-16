// Function: sub_20F2770
// Address: 0x20f2770
//
unsigned __int64 __fastcall sub_20F2770(__int64 a1, __int64 a2, __int32 a3)
{
  unsigned __int64 result; // rax
  unsigned __int32 v7; // esi
  __int64 v8; // rdx
  _QWORD *v9; // rcx
  int v10; // edi
  unsigned __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rsi
  unsigned int v15; // r8d
  __int64 *v16; // rax
  __int64 v17; // r10
  __int64 v18; // r15
  unsigned __int64 v19; // r14
  __int64 *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // r12
  _QWORD *v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rcx
  int v26; // ecx
  int v27; // r9d
  int v28; // eax
  int v29; // r11d
  __m128i v30; // [rsp-48h] [rbp-48h] BYREF

  result = *(unsigned int *)(a1 + 272);
  if ( !(_DWORD)result )
    return result;
  v7 = (result - 1) & (37 * a3);
  v8 = *(_QWORD *)(a1 + 256);
  v9 = (_QWORD *)(v8 + 16LL * v7);
  v10 = *(_DWORD *)v9;
  if ( *(_DWORD *)v9 != a3 )
  {
    v26 = 1;
    while ( v10 != 0x7FFFFFFF )
    {
      v27 = v26 + 1;
      v7 = (result - 1) & (v26 + v7);
      v9 = (_QWORD *)(v8 + 16LL * v7);
      v10 = *(_DWORD *)v9;
      if ( *(_DWORD *)v9 == a3 )
        goto LABEL_3;
      v26 = v27;
    }
    return result;
  }
LABEL_3:
  result = v8 + 16 * result;
  if ( v9 == (_QWORD *)result )
    return result;
  v11 = a2;
  v12 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 272LL);
  if ( (*(_BYTE *)(a2 + 46) & 4) != 0 )
  {
    do
      v11 = *(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL;
    while ( (*(_BYTE *)(v11 + 46) & 4) != 0 );
  }
  v13 = *(_QWORD *)(v12 + 368);
  v14 = *(unsigned int *)(v12 + 384);
  if ( (_DWORD)v14 )
  {
    v15 = (v14 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
    v16 = (__int64 *)(v13 + 16LL * v15);
    v17 = *v16;
    if ( *v16 == v11 )
      goto LABEL_8;
    v28 = 1;
    while ( v17 != -8 )
    {
      v29 = v28 + 1;
      v15 = (v14 - 1) & (v28 + v15);
      v16 = (__int64 *)(v13 + 16LL * v15);
      v17 = *v16;
      if ( *v16 == v11 )
        goto LABEL_8;
      v28 = v29;
    }
  }
  v16 = (__int64 *)(v13 + 16 * v14);
LABEL_8:
  v18 = v9[1];
  v19 = v16[1] & 0xFFFFFFFFFFFFFFF8LL;
  v20 = (__int64 *)sub_1DB3C70((__int64 *)v18, v19 | 4);
  if ( v20 == (__int64 *)(*(_QWORD *)v18 + 24LL * *(unsigned int *)(v18 + 8))
    || (*(_DWORD *)((*v20 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v20 >> 1) & 3) > (*(_DWORD *)(v19 + 24) | 2u) )
  {
    v21 = 0;
  }
  else
  {
    v21 = v20[2];
  }
  v30.m128i_i32[0] = a3;
  v30.m128i_i64[1] = v21;
  v22 = sub_20F1860(a1 + 280, &v30);
  result = *(_QWORD *)(v22 + 8);
  if ( *(_QWORD *)(v22 + 16) == result )
  {
    v23 = (_QWORD *)(result + 8LL * *(unsigned int *)(v22 + 28));
    if ( (_QWORD *)result == v23 )
    {
LABEL_28:
      result = (unsigned __int64)v23;
    }
    else
    {
      while ( a2 != *(_QWORD *)result )
      {
        result += 8LL;
        if ( v23 == (_QWORD *)result )
          goto LABEL_28;
      }
    }
  }
  else
  {
    result = (unsigned __int64)sub_16CC9F0(v22, a2);
    if ( a2 == *(_QWORD *)result )
    {
      v24 = *(_QWORD *)(v22 + 16);
      if ( v24 == *(_QWORD *)(v22 + 8) )
        v25 = *(unsigned int *)(v22 + 28);
      else
        v25 = *(unsigned int *)(v22 + 24);
      v23 = (_QWORD *)(v24 + 8 * v25);
    }
    else
    {
      result = *(_QWORD *)(v22 + 16);
      if ( result != *(_QWORD *)(v22 + 8) )
        return result;
      result += 8LL * *(unsigned int *)(v22 + 28);
      v23 = (_QWORD *)result;
    }
  }
  if ( v23 != (_QWORD *)result )
  {
    *(_QWORD *)result = -2;
    ++*(_DWORD *)(v22 + 32);
  }
  return result;
}
