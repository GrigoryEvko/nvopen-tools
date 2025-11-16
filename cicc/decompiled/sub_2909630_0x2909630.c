// Function: sub_2909630
// Address: 0x2909630
//
__int64 __fastcall sub_2909630(__int64 a1, __int64 *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r8
  __int64 *v6; // r10
  int v7; // r11d
  unsigned int v8; // eax
  __int64 *v9; // rdi
  __int64 v10; // rcx
  int v12; // eax
  int v13; // edx
  __int64 *v14; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    v14 = 0;
LABEL_17:
    v4 *= 2;
    goto LABEL_18;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = 0;
  v7 = 1;
  v8 = (v4 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v9 = (__int64 *)(v5 + 8LL * v8);
  v10 = *v9;
  if ( *a2 == *v9 )
    return 0;
  while ( v10 != -4096 )
  {
    if ( v6 || v10 != -8192 )
      v9 = v6;
    v8 = (v4 - 1) & (v7 + v8);
    v10 = *(_QWORD *)(v5 + 8LL * v8);
    if ( *a2 == v10 )
      return 0;
    ++v7;
    v6 = v9;
    v9 = (__int64 *)(v5 + 8LL * v8);
  }
  v12 = *(_DWORD *)(a1 + 16);
  if ( !v6 )
    v6 = v9;
  ++*(_QWORD *)a1;
  v13 = v12 + 1;
  v14 = v6;
  if ( 4 * (v12 + 1) >= 3 * v4 )
    goto LABEL_17;
  if ( v4 - *(_DWORD *)(a1 + 20) - v13 <= v4 >> 3 )
  {
LABEL_18:
    sub_CE2A30(a1, v4);
    sub_DA5B20(a1, a2, &v14);
    v6 = v14;
    v13 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v6 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v6 = *a2;
  sub_94F890(a1 + 32, *a2);
  return 1;
}
