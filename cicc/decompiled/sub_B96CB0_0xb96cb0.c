// Function: sub_B96CB0
// Address: 0xb96cb0
//
__int64 __fastcall sub_B96CB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  char v5; // di
  int v6; // edi
  __int64 v7; // r9
  int v8; // r8d
  unsigned int v9; // edx
  _QWORD *v10; // r10
  __int64 v11; // rcx
  __int64 result; // rax
  unsigned int v13; // r8d
  unsigned int v14; // eax
  int v15; // edx
  unsigned int v16; // edi
  __int64 *v17; // rax
  int v18; // r12d
  _QWORD *v19; // r11
  unsigned int v20; // esi
  __int64 *v21; // [rsp+8h] [rbp-48h] BYREF
  __int64 v22; // [rsp+10h] [rbp-40h] BYREF
  __m128i v23; // [rsp+18h] [rbp-38h] BYREF

  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_BYTE *)(a1 + 24);
  v22 = a2;
  v23.m128i_i64[0] = a3;
  v23.m128i_i64[1] = v4;
  v6 = v5 & 1;
  if ( v6 )
  {
    v7 = a1 + 32;
    v8 = 3;
  }
  else
  {
    v13 = *(_DWORD *)(a1 + 40);
    v7 = *(_QWORD *)(a1 + 32);
    if ( !v13 )
    {
      v14 = *(_DWORD *)(a1 + 24);
      ++*(_QWORD *)(a1 + 16);
      v21 = 0;
      v15 = (v14 >> 1) + 1;
LABEL_8:
      v16 = 3 * v13;
      goto LABEL_9;
    }
    v8 = v13 - 1;
  }
  v9 = v8 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (_QWORD *)(v7 + 24LL * v9);
  v11 = *v10;
  if ( a2 == *v10 )
    goto LABEL_4;
  v18 = 1;
  v19 = 0;
  while ( v11 != -4096 )
  {
    if ( v19 || v11 != -8192 )
      v10 = v19;
    v9 = v8 & (v18 + v9);
    v11 = *(_QWORD *)(v7 + 24LL * v9);
    if ( a2 == v11 )
      goto LABEL_4;
    ++v18;
    v19 = v10;
    v10 = (_QWORD *)(v7 + 24LL * v9);
  }
  v14 = *(_DWORD *)(a1 + 24);
  if ( !v19 )
    v19 = v10;
  ++*(_QWORD *)(a1 + 16);
  v21 = v19;
  v15 = (v14 >> 1) + 1;
  if ( !(_BYTE)v6 )
  {
    v13 = *(_DWORD *)(a1 + 40);
    goto LABEL_8;
  }
  v16 = 12;
  v13 = 4;
LABEL_9:
  if ( v16 <= 4 * v15 )
  {
    v20 = 2 * v13;
LABEL_21:
    sub_B95E60(a1 + 16, v20);
    sub_B926F0(a1 + 16, &v22, &v21);
    a2 = v22;
    v14 = *(_DWORD *)(a1 + 24);
    goto LABEL_11;
  }
  if ( v13 - *(_DWORD *)(a1 + 28) - v15 <= v13 >> 3 )
  {
    v20 = v13;
    goto LABEL_21;
  }
LABEL_11:
  *(_DWORD *)(a1 + 24) = (2 * (v14 >> 1) + 2) | v14 & 1;
  v17 = v21;
  if ( *v21 != -4096 )
    --*(_DWORD *)(a1 + 28);
  *v17 = a2;
  *(__m128i *)(v17 + 1) = _mm_loadu_si128(&v23);
  v4 = *(_QWORD *)(a1 + 8);
LABEL_4:
  result = v4 + 1;
  *(_QWORD *)(a1 + 8) = result;
  return result;
}
