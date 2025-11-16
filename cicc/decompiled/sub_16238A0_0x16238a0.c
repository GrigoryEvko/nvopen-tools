// Function: sub_16238A0
// Address: 0x16238a0
//
__int64 __fastcall sub_16238A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  char v5; // cl
  int v6; // ecx
  __int64 v7; // r9
  int v8; // r8d
  unsigned int v9; // edx
  __int64 *v10; // r10
  __int64 v11; // rdi
  __int64 result; // rax
  unsigned int v13; // r8d
  unsigned int v14; // eax
  __int64 *v15; // r11
  int v16; // edx
  unsigned int v17; // edi
  int v18; // r12d
  unsigned int v19; // esi
  __int64 *v20; // [rsp+8h] [rbp-48h] BYREF
  __int64 v21; // [rsp+10h] [rbp-40h] BYREF
  __m128i v22; // [rsp+18h] [rbp-38h] BYREF

  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_BYTE *)(a1 + 24);
  v21 = a2;
  v22.m128i_i64[0] = a3;
  v22.m128i_i64[1] = v4;
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
      v15 = 0;
      v16 = (v14 >> 1) + 1;
LABEL_8:
      v17 = 3 * v13;
      goto LABEL_9;
    }
    v8 = v13 - 1;
  }
  v9 = v8 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v7 + 24LL * v9);
  v11 = *v10;
  if ( a2 == *v10 )
    goto LABEL_4;
  v18 = 1;
  v15 = 0;
  while ( v11 != -4 )
  {
    if ( v15 || v11 != -8 )
      v10 = v15;
    v9 = v8 & (v18 + v9);
    v11 = *(_QWORD *)(v7 + 24LL * v9);
    if ( a2 == v11 )
      goto LABEL_4;
    ++v18;
    v15 = v10;
    v10 = (__int64 *)(v7 + 24LL * v9);
  }
  v14 = *(_DWORD *)(a1 + 24);
  if ( !v15 )
    v15 = v10;
  ++*(_QWORD *)(a1 + 16);
  v16 = (v14 >> 1) + 1;
  if ( !(_BYTE)v6 )
  {
    v13 = *(_DWORD *)(a1 + 40);
    goto LABEL_8;
  }
  v17 = 12;
  v13 = 4;
LABEL_9:
  if ( 4 * v16 >= v17 )
  {
    v19 = 2 * v13;
LABEL_21:
    sub_1622AB0(a1 + 16, v19);
    sub_1621140(a1 + 16, &v21, &v20);
    v15 = v20;
    a2 = v21;
    v14 = *(_DWORD *)(a1 + 24);
    goto LABEL_11;
  }
  if ( v13 - *(_DWORD *)(a1 + 28) - v16 <= v13 >> 3 )
  {
    v19 = v13;
    goto LABEL_21;
  }
LABEL_11:
  *(_DWORD *)(a1 + 24) = (2 * (v14 >> 1) + 2) | v14 & 1;
  if ( *v15 != -4 )
    --*(_DWORD *)(a1 + 28);
  *v15 = a2;
  *(__m128i *)(v15 + 1) = _mm_loadu_si128(&v22);
  v4 = *(_QWORD *)(a1 + 8);
LABEL_4:
  result = v4 + 1;
  *(_QWORD *)(a1 + 8) = result;
  return result;
}
