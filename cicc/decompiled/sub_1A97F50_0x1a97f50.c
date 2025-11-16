// Function: sub_1A97F50
// Address: 0x1a97f50
//
__int64 __fastcall sub_1A97F50(__int64 a1, __int64 *a2)
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
  __int64 v14; // rax
  _BYTE *v15; // rsi
  __int64 *v16; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
LABEL_20:
    v4 *= 2;
    goto LABEL_21;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = 0;
  v7 = 1;
  v8 = (v4 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v9 = (__int64 *)(v5 + 8LL * v8);
  v10 = *v9;
  if ( *a2 == *v9 )
    return 0;
  while ( v10 != -8 )
  {
    if ( v10 != -16 || v6 )
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
  if ( 4 * (v12 + 1) >= 3 * v4 )
    goto LABEL_20;
  if ( v4 - *(_DWORD *)(a1 + 20) - v13 <= v4 >> 3 )
  {
LABEL_21:
    sub_1353F00(a1, v4);
    sub_1A97120(a1, a2, &v16);
    v6 = v16;
    v13 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v6 != -8 )
    --*(_DWORD *)(a1 + 20);
  v14 = *a2;
  *v6 = *a2;
  v15 = *(_BYTE **)(a1 + 40);
  if ( v15 == *(_BYTE **)(a1 + 48) )
  {
    sub_1287830(a1 + 32, v15, a2);
    return 1;
  }
  else
  {
    if ( v15 )
    {
      *(_QWORD *)v15 = v14;
      v15 = *(_BYTE **)(a1 + 40);
    }
    *(_QWORD *)(a1 + 40) = v15 + 8;
    return 1;
  }
}
