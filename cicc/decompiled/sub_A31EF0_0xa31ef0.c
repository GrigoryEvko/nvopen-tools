// Function: sub_A31EF0
// Address: 0xa31ef0
//
__int64 __fastcall sub_A31EF0(__int64 *a1, int *a2)
{
  __int64 v2; // rbx
  int v3; // eax
  unsigned int v4; // esi
  __int64 v5; // r9
  int *v6; // rdi
  int v7; // r11d
  unsigned int v8; // edx
  int *v9; // rcx
  int v10; // r8d
  int v12; // ecx
  int v13; // ecx
  int v14; // [rsp+4h] [rbp-2Ch] BYREF
  int *v15; // [rsp+8h] [rbp-28h] BYREF

  v2 = *a1;
  v3 = *a2;
  v4 = *(_DWORD *)(*a1 + 136);
  v14 = v3;
  if ( !v4 )
  {
    ++*(_QWORD *)(v2 + 112);
    v15 = 0;
LABEL_18:
    v4 *= 2;
    goto LABEL_19;
  }
  v5 = *(_QWORD *)(v2 + 120);
  v6 = 0;
  v7 = 1;
  v8 = (v4 - 1) & (37 * v3);
  v9 = (int *)(v5 + 8LL * v8);
  v10 = *v9;
  if ( v3 == *v9 )
    return (unsigned int)v9[1];
  while ( v10 != -1 )
  {
    if ( !v6 && v10 == -2 )
      v6 = v9;
    v8 = (v4 - 1) & (v7 + v8);
    v9 = (int *)(v5 + 8LL * v8);
    v10 = *v9;
    if ( v3 == *v9 )
      return (unsigned int)v9[1];
    ++v7;
  }
  if ( !v6 )
    v6 = v9;
  v12 = *(_DWORD *)(v2 + 128);
  ++*(_QWORD *)(v2 + 112);
  v13 = v12 + 1;
  v15 = v6;
  if ( 4 * v13 >= 3 * v4 )
    goto LABEL_18;
  if ( v4 - *(_DWORD *)(v2 + 132) - v13 <= v4 >> 3 )
  {
LABEL_19:
    sub_A09770(v2 + 112, v4);
    sub_A1A0F0(v2 + 112, &v14, &v15);
    v3 = v14;
    v6 = v15;
    v13 = *(_DWORD *)(v2 + 128) + 1;
  }
  *(_DWORD *)(v2 + 128) = v13;
  if ( *v6 != -1 )
    --*(_DWORD *)(v2 + 132);
  *v6 = v3;
  v6[1] = 0;
  return 0;
}
