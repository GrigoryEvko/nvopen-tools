// Function: sub_256A430
// Address: 0x256a430
//
__int64 *__fastcall sub_256A430(__int64 a1, __int64 *a2)
{
  unsigned int v3; // esi
  __int64 v4; // r9
  int v5; // r11d
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 *v8; // rax
  __int64 v9; // r8
  int v11; // eax
  int v12; // ecx
  __int64 *v13; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    ++*(_QWORD *)a1;
    v13 = 0;
LABEL_18:
    v3 *= 2;
    goto LABEL_19;
  }
  v4 = *(_QWORD *)(a1 + 8);
  v5 = 1;
  v6 = (v3 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v7 = 0;
  v8 = (__int64 *)(v4 + 104LL * v6);
  v9 = *v8;
  if ( *a2 == *v8 )
    return v8 + 1;
  while ( v9 != -4096 )
  {
    if ( v9 == -8192 && !v7 )
      v7 = v8;
    v6 = (v3 - 1) & (v5 + v6);
    v8 = (__int64 *)(v4 + 104LL * v6);
    v9 = *v8;
    if ( *a2 == *v8 )
      return v8 + 1;
    ++v5;
  }
  if ( !v7 )
    v7 = v8;
  v11 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v12 = v11 + 1;
  v13 = v7;
  if ( 4 * (v11 + 1) >= 3 * v3 )
    goto LABEL_18;
  if ( v3 - *(_DWORD *)(a1 + 20) - v12 <= v3 >> 3 )
  {
LABEL_19:
    sub_2569FF0(a1, v3);
    sub_255E0C0(a1, a2, &v13);
    v7 = v13;
    v12 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v12;
  if ( *v7 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v7 = *a2;
  memset(v7 + 1, 0, 0x60u);
  v7[1] = (__int64)(v7 + 3);
  v7[2] = 0x400000000LL;
  v7[10] = (__int64)(v7 + 8);
  v7[11] = (__int64)(v7 + 8);
  return v7 + 1;
}
