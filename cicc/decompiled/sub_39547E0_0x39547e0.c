// Function: sub_39547E0
// Address: 0x39547e0
//
__int64 __fastcall sub_39547E0(__int64 a1, __int64 a2)
{
  int v4; // eax
  int v5; // edx
  __int64 v6; // rsi
  unsigned int v7; // eax
  __int64 v8; // rcx
  int v10; // edi
  _BYTE *v11; // rsi
  _BYTE *v12; // rsi
  int v13; // r12d
  unsigned int v14; // esi
  __int64 v15; // rcx
  __int64 v16; // r8
  unsigned int v17; // edx
  __int64 *v18; // rax
  __int64 v19; // rdi
  int v20; // r11d
  __int64 *v21; // r10
  int v22; // edi
  int v23; // edi
  __int64 v24[2]; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v25[5]; // [rsp+18h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 80);
  v24[0] = a2;
  if ( v4 )
  {
    v5 = v4 - 1;
    v6 = *(_QWORD *)(a1 + 64);
    v7 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = *(_QWORD *)(v6 + 16LL * v7);
    if ( a2 == v8 )
      return 0;
    v10 = 1;
    while ( v8 != -8 )
    {
      v7 = v5 & (v10 + v7);
      v8 = *(_QWORD *)(v6 + 16LL * v7);
      if ( a2 == v8 )
        return 0;
      ++v10;
    }
  }
  if ( *(_BYTE *)(a2 + 16) == 17 && sub_3953740(a1 + 176, a2) )
    return 0;
  v11 = *(_BYTE **)(a1 + 96);
  if ( v11 == *(_BYTE **)(a1 + 104) )
  {
    sub_1287830(a1 + 88, v11, v24);
    v12 = *(_BYTE **)(a1 + 96);
  }
  else
  {
    if ( v11 )
    {
      *(_QWORD *)v11 = v24[0];
      v11 = *(_BYTE **)(a1 + 96);
    }
    v12 = v11 + 8;
    *(_QWORD *)(a1 + 96) = v12;
  }
  v13 = ((__int64)&v12[-*(_QWORD *)(a1 + 88)] >> 3) - 1;
  v14 = *(_DWORD *)(a1 + 80);
  if ( !v14 )
  {
    ++*(_QWORD *)(a1 + 56);
    goto LABEL_29;
  }
  v15 = v24[0];
  v16 = *(_QWORD *)(a1 + 64);
  v17 = (v14 - 1) & ((LODWORD(v24[0]) >> 9) ^ (LODWORD(v24[0]) >> 4));
  v18 = (__int64 *)(v16 + 16LL * v17);
  v19 = *v18;
  if ( *v18 != v24[0] )
  {
    v20 = 1;
    v21 = 0;
    while ( v19 != -8 )
    {
      if ( v19 == -16 && !v21 )
        v21 = v18;
      v17 = (v14 - 1) & (v20 + v17);
      v18 = (__int64 *)(v16 + 16LL * v17);
      v19 = *v18;
      if ( v24[0] == *v18 )
        goto LABEL_13;
      ++v20;
    }
    v22 = *(_DWORD *)(a1 + 72);
    if ( v21 )
      v18 = v21;
    ++*(_QWORD *)(a1 + 56);
    v23 = v22 + 1;
    if ( 4 * v23 < 3 * v14 )
    {
      if ( v14 - *(_DWORD *)(a1 + 76) - v23 > v14 >> 3 )
      {
LABEL_25:
        *(_DWORD *)(a1 + 72) = v23;
        if ( *v18 != -8 )
          --*(_DWORD *)(a1 + 76);
        *v18 = v15;
        *((_DWORD *)v18 + 2) = 0;
        goto LABEL_13;
      }
LABEL_30:
      sub_1BFE340(a1 + 56, v14);
      sub_1BFD9C0(a1 + 56, v24, v25);
      v18 = (__int64 *)v25[0];
      v15 = v24[0];
      v23 = *(_DWORD *)(a1 + 72) + 1;
      goto LABEL_25;
    }
LABEL_29:
    v14 *= 2;
    goto LABEL_30;
  }
LABEL_13:
  *((_DWORD *)v18 + 2) = v13;
  return 1;
}
