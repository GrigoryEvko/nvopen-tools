// Function: sub_1CA8350
// Address: 0x1ca8350
//
__int64 __fastcall sub_1CA8350(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rsi
  unsigned int v9; // ecx
  __int64 v10; // rdx
  _BYTE *v11; // r11
  int v13; // edx
  unsigned int v14; // eax
  unsigned int v15; // esi
  unsigned int v16; // r12d
  __int64 v17; // rcx
  __int64 v18; // r8
  unsigned int v19; // edx
  __int64 v20; // rax
  _BYTE *v21; // rdi
  int v22; // r12d
  int v23; // r11d
  __int64 v24; // r10
  int v25; // edi
  int v26; // edi
  _BYTE *v27; // [rsp+8h] [rbp-38h] BYREF
  __int64 v28[5]; // [rsp+18h] [rbp-28h] BYREF

  v7 = *(unsigned int *)(a3 + 24);
  v27 = a2;
  if ( (_DWORD)v7 )
  {
    v8 = *(_QWORD *)(a3 + 8);
    v9 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v10 = v8 + 16LL * v9;
    v11 = *(_BYTE **)v10;
    if ( a2 == *(_BYTE **)v10 )
    {
LABEL_3:
      if ( v10 != v8 + 16 * v7 )
        return *(unsigned int *)(v10 + 8);
    }
    else
    {
      v13 = 1;
      while ( v11 != (_BYTE *)-8LL )
      {
        v22 = v13 + 1;
        v9 = (v7 - 1) & (v13 + v9);
        v10 = v8 + 16LL * v9;
        v11 = *(_BYTE **)v10;
        if ( a2 == *(_BYTE **)v10 )
          goto LABEL_3;
        v13 = v22;
      }
    }
  }
  v14 = sub_1CA7E20(a1, a2, a3, a4);
  v15 = *(_DWORD *)(a3 + 24);
  v16 = v14;
  if ( !v15 )
  {
    ++*(_QWORD *)a3;
    goto LABEL_22;
  }
  v17 = (__int64)v27;
  v18 = *(_QWORD *)(a3 + 8);
  v19 = (v15 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
  v20 = v18 + 16LL * v19;
  v21 = *(_BYTE **)v20;
  if ( *(_BYTE **)v20 != v27 )
  {
    v23 = 1;
    v24 = 0;
    while ( v21 != (_BYTE *)-8LL )
    {
      if ( v21 == (_BYTE *)-16LL && !v24 )
        v24 = v20;
      v19 = (v15 - 1) & (v23 + v19);
      v20 = v18 + 16LL * v19;
      v21 = *(_BYTE **)v20;
      if ( v27 == *(_BYTE **)v20 )
        goto LABEL_9;
      ++v23;
    }
    v25 = *(_DWORD *)(a3 + 16);
    if ( v24 )
      v20 = v24;
    ++*(_QWORD *)a3;
    v26 = v25 + 1;
    if ( 4 * v26 < 3 * v15 )
    {
      if ( v15 - *(_DWORD *)(a3 + 20) - v26 > v15 >> 3 )
      {
LABEL_18:
        *(_DWORD *)(a3 + 16) = v26;
        if ( *(_QWORD *)v20 != -8 )
          --*(_DWORD *)(a3 + 20);
        *(_QWORD *)v20 = v17;
        *(_DWORD *)(v20 + 8) = 0;
        goto LABEL_9;
      }
LABEL_23:
      sub_177C7D0(a3, v15);
      sub_190E590(a3, (__int64 *)&v27, v28);
      v20 = v28[0];
      v17 = (__int64)v27;
      v26 = *(_DWORD *)(a3 + 16) + 1;
      goto LABEL_18;
    }
LABEL_22:
    v15 *= 2;
    goto LABEL_23;
  }
LABEL_9:
  *(_DWORD *)(v20 + 8) = v16;
  return v16;
}
