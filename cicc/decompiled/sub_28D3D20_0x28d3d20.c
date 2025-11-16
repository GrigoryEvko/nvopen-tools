// Function: sub_28D3D20
// Address: 0x28d3d20
//
_QWORD *__fastcall sub_28D3D20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  unsigned int v6; // esi
  __int64 v7; // rcx
  __int64 v8; // r8
  int v9; // r11d
  unsigned int v10; // edx
  __int64 *v11; // rdi
  __int64 *v12; // rax
  __int64 v13; // r10
  __int64 v14; // rdi
  int v16; // edi
  int v17; // edi
  __int64 v18; // [rsp+8h] [rbp-38h] BYREF
  __int64 *v19; // [rsp+18h] [rbp-28h] BYREF

  v3 = a1 + 1888;
  v18 = a2;
  v6 = *(_DWORD *)(a1 + 1912);
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 1888);
    v19 = 0;
LABEL_19:
    v6 *= 2;
    goto LABEL_20;
  }
  v7 = v18;
  v8 = *(_QWORD *)(a1 + 1896);
  v9 = 1;
  v10 = (v6 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
  v11 = (__int64 *)(v8 + 56LL * v10);
  v12 = 0;
  v13 = *v11;
  if ( v18 == *v11 )
  {
LABEL_3:
    v14 = (__int64)(v11 + 1);
    return sub_AE6EC0(v14, a3);
  }
  while ( v13 != -4096 )
  {
    if ( !v12 && v13 == -8192 )
      v12 = v11;
    v10 = (v6 - 1) & (v9 + v10);
    v11 = (__int64 *)(v8 + 56LL * v10);
    v13 = *v11;
    if ( v18 == *v11 )
      goto LABEL_3;
    ++v9;
  }
  if ( !v12 )
    v12 = v11;
  v16 = *(_DWORD *)(a1 + 1904);
  ++*(_QWORD *)(a1 + 1888);
  v17 = v16 + 1;
  v19 = v12;
  if ( 4 * v17 >= 3 * v6 )
    goto LABEL_19;
  if ( v6 - *(_DWORD *)(a1 + 1908) - v17 <= v6 >> 3 )
  {
LABEL_20:
    sub_28D3B00(v3, v6);
    sub_28CBC50(v3, &v18, &v19);
    v7 = v18;
    v17 = *(_DWORD *)(a1 + 1904) + 1;
    v12 = v19;
  }
  *(_DWORD *)(a1 + 1904) = v17;
  if ( *v12 != -4096 )
    --*(_DWORD *)(a1 + 1908);
  *v12 = v7;
  v14 = (__int64)(v12 + 1);
  v12[1] = 0;
  v12[2] = (__int64)(v12 + 5);
  v12[3] = 2;
  *((_DWORD *)v12 + 8) = 0;
  *((_BYTE *)v12 + 36) = 1;
  return sub_AE6EC0(v14, a3);
}
