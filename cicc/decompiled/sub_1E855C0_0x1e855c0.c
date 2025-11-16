// Function: sub_1E855C0
// Address: 0x1e855c0
//
__int64 *__fastcall sub_1E855C0(__int64 a1, __int64 *a2)
{
  unsigned int v4; // esi
  __int64 v5; // rdi
  unsigned int v6; // eax
  __int64 *v7; // r8
  __int64 v8; // rcx
  int v10; // r11d
  __int64 *v11; // r10
  int v12; // eax
  int v13; // edx
  __int64 *v14; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
LABEL_14:
    v4 *= 2;
    goto LABEL_15;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (v4 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v7 = (__int64 *)(v5 + 384LL * v6);
  v8 = *v7;
  if ( *a2 == *v7 )
    return v7;
  v10 = 1;
  v11 = 0;
  while ( v8 != -8 )
  {
    if ( v8 == -16 && !v11 )
      v11 = v7;
    v6 = (v4 - 1) & (v10 + v6);
    v7 = (__int64 *)(v5 + 384LL * v6);
    v8 = *v7;
    if ( *a2 == *v7 )
      return v7;
    ++v10;
  }
  v12 = *(_DWORD *)(a1 + 16);
  if ( v11 )
    v7 = v11;
  ++*(_QWORD *)a1;
  v13 = v12 + 1;
  if ( 4 * (v12 + 1) >= 3 * v4 )
    goto LABEL_14;
  if ( v4 - *(_DWORD *)(a1 + 20) - v13 <= v4 >> 3 )
  {
LABEL_15:
    sub_1E850A0(a1, v4);
    sub_1E84D20(a1, a2, &v14);
    v7 = v14;
    v13 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v7 != -8 )
    --*(_DWORD *)(a1 + 20);
  *v7 = *a2;
  memset(v7 + 1, 0, 0x178u);
  v7[23] = (__int64)(v7 + 27);
  v7[36] = (__int64)(v7 + 40);
  v7[37] = (__int64)(v7 + 40);
  v7[24] = (__int64)(v7 + 27);
  v7[25] = 8;
  v7[38] = 8;
  return v7;
}
