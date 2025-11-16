// Function: sub_154E370
// Address: 0x154e370
//
__int64 *__fastcall sub_154E370(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  int v4; // r12d
  unsigned int v5; // esi
  __int64 v6; // rcx
  __int64 v7; // r8
  unsigned int v8; // edx
  __int64 *result; // rax
  __int64 v10; // rdi
  int v11; // r11d
  __int64 *v12; // r10
  int v13; // edi
  int v14; // edi
  __int64 v15; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v16[5]; // [rsp+18h] [rbp-28h] BYREF

  v2 = a1 + 72;
  v4 = *(_DWORD *)(a1 + 104);
  v15 = a2;
  v5 = *(_DWORD *)(a1 + 96);
  *(_DWORD *)(a1 + 104) = v4 + 1;
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 72);
LABEL_14:
    v5 *= 2;
    goto LABEL_15;
  }
  v6 = v15;
  v7 = *(_QWORD *)(a1 + 80);
  v8 = (v5 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
  result = (__int64 *)(v7 + 16LL * v8);
  v10 = *result;
  if ( v15 == *result )
    goto LABEL_3;
  v11 = 1;
  v12 = 0;
  while ( v10 != -8 )
  {
    if ( v10 == -16 && !v12 )
      v12 = result;
    v8 = (v5 - 1) & (v11 + v8);
    result = (__int64 *)(v7 + 16LL * v8);
    v10 = *result;
    if ( v15 == *result )
      goto LABEL_3;
    ++v11;
  }
  v13 = *(_DWORD *)(a1 + 88);
  if ( v12 )
    result = v12;
  ++*(_QWORD *)(a1 + 72);
  v14 = v13 + 1;
  if ( 4 * v14 >= 3 * v5 )
    goto LABEL_14;
  if ( v5 - *(_DWORD *)(a1 + 92) - v14 <= v5 >> 3 )
  {
LABEL_15:
    sub_1542080(v2, v5);
    sub_154CC80(v2, &v15, v16);
    result = (__int64 *)v16[0];
    v6 = v15;
    v14 = *(_DWORD *)(a1 + 88) + 1;
  }
  *(_DWORD *)(a1 + 88) = v14;
  if ( *result != -8 )
    --*(_DWORD *)(a1 + 92);
  *result = v6;
  *((_DWORD *)result + 2) = 0;
LABEL_3:
  *((_DWORD *)result + 2) = v4;
  return result;
}
