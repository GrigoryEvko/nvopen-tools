// Function: sub_2B4B450
// Address: 0x2b4b450
//
__int64 *__fastcall sub_2B4B450(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // r12
  __int64 v5; // r13
  __int64 v7; // rax
  _QWORD *v8; // rdx
  __int64 *v9; // r8
  __int64 *v10; // r9
  __int64 *v11; // r13
  _BYTE *v12; // rsi
  __int64 v13; // r10
  int v14; // r11d
  __int64 v15; // rcx
  int v16; // eax
  unsigned int v17; // edi
  _BYTE *v18; // r14
  _BYTE *v20; // rsi
  int v21; // eax
  unsigned int v22; // edi
  _BYTE *v23; // r10
  _BYTE *v24; // rdi
  int v25; // eax
  unsigned int v26; // esi
  _BYTE *v27; // r10
  int v28; // r11d
  _BYTE *v29; // rdi
  int v30; // eax
  unsigned int v31; // esi
  _BYTE *v32; // r10
  int v33; // r11d
  int v34; // r11d
  int v35; // [rsp+Ch] [rbp-44h]
  __int64 v36[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = a1;
  v5 = (a2 - (__int64)a1) >> 5;
  v7 = (a2 - (__int64)a1) >> 3;
  if ( v5 > 0 )
  {
    v8 = a1 + 2;
    v9 = a1 + 1;
    v10 = a1 + 3;
    v11 = &a1[4 * v5];
    while ( 1 )
    {
      v12 = (_BYTE *)*(v8 - 2);
      if ( *v12 > 0x1Cu && (v13 = *(_QWORD *)(a3 + 1984), v14 = *(_DWORD *)(a3 + 2000), v15 = v13, v14) )
      {
        v16 = v14 - 1;
        v17 = (v14 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v18 = *(_BYTE **)(v13 + 8LL * v17);
        if ( v12 == v18 )
          return v4;
        v35 = 1;
        while ( v18 != (_BYTE *)-4096LL )
        {
          v17 = v16 & (v35 + v17);
          ++v35;
          v18 = *(_BYTE **)(v13 + 8LL * v17);
          if ( v12 == v18 )
            return v4;
        }
        v20 = (_BYTE *)*(v8 - 1);
        if ( *v20 <= 0x1Cu )
        {
          v24 = (_BYTE *)*v8;
          if ( *(_BYTE *)*v8 <= 0x1Cu )
            goto LABEL_27;
          v15 = *(_QWORD *)(a3 + 1984);
          v25 = *(_DWORD *)(a3 + 2000);
          goto LABEL_14;
        }
      }
      else
      {
        v20 = (_BYTE *)*(v8 - 1);
        if ( *v20 <= 0x1Cu )
        {
          v24 = (_BYTE *)*v8;
          if ( *(_BYTE *)*v8 <= 0x1Cu )
            goto LABEL_27;
          v25 = *(_DWORD *)(a3 + 2000);
          v15 = *(_QWORD *)(a3 + 1984);
          if ( !v25 )
            goto LABEL_18;
LABEL_14:
          v16 = v25 - 1;
          goto LABEL_15;
        }
        v21 = *(_DWORD *)(a3 + 2000);
        v15 = *(_QWORD *)(a3 + 1984);
        if ( !v21 )
        {
          if ( *(_BYTE *)*v8 > 0x1Cu )
            goto LABEL_18;
LABEL_27:
          v29 = (_BYTE *)v8[1];
          if ( *v29 <= 0x1Cu )
            goto LABEL_18;
          v30 = *(_DWORD *)(a3 + 2000);
          v15 = *(_QWORD *)(a3 + 1984);
          if ( !v30 )
            goto LABEL_18;
          v16 = v30 - 1;
          goto LABEL_30;
        }
        v16 = v21 - 1;
      }
      v22 = v16 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v23 = *(_BYTE **)(v15 + 8LL * v22);
      if ( v20 == v23 )
        return v9;
      v28 = 1;
      while ( v23 != (_BYTE *)-4096LL )
      {
        v22 = v16 & (v28 + v22);
        v23 = *(_BYTE **)(v15 + 8LL * v22);
        if ( v23 == v20 )
          return v9;
        ++v28;
      }
      v24 = (_BYTE *)*v8;
      if ( *(_BYTE *)*v8 <= 0x1Cu )
        goto LABEL_27;
LABEL_15:
      v26 = v16 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v27 = *(_BYTE **)(v15 + 8LL * v26);
      if ( v27 == v24 )
        return v8;
      v33 = 1;
      while ( v27 != (_BYTE *)-4096LL )
      {
        v26 = v16 & (v33 + v26);
        v27 = *(_BYTE **)(v15 + 8LL * v26);
        if ( v27 == v24 )
          return v8;
        ++v33;
      }
      v29 = (_BYTE *)v8[1];
      if ( *v29 <= 0x1Cu )
        goto LABEL_18;
LABEL_30:
      v31 = v16 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
      v32 = *(_BYTE **)(v15 + 8LL * v31);
      if ( v29 == v32 )
        return v10;
      v34 = 1;
      while ( v32 != (_BYTE *)-4096LL )
      {
        v31 = v16 & (v34 + v31);
        v32 = *(_BYTE **)(v15 + 8LL * v31);
        if ( v29 == v32 )
          return v10;
        ++v34;
      }
LABEL_18:
      v4 += 4;
      v8 += 4;
      v9 += 4;
      v10 += 4;
      if ( v11 == v4 )
      {
        v7 = (a2 - (__int64)v4) >> 3;
        break;
      }
    }
  }
  if ( v7 == 2 )
  {
LABEL_48:
    if ( *(_BYTE *)*v4 > 0x1Cu )
    {
      v36[0] = *v4;
      if ( sub_2B4B3F0(a3 + 1976, v36) )
        return v4;
    }
    ++v4;
    goto LABEL_51;
  }
  if ( v7 == 3 )
  {
    if ( *(_BYTE *)*v4 > 0x1Cu )
    {
      v36[0] = *v4;
      if ( sub_2B4B3F0(a3 + 1976, v36) )
        return v4;
    }
    ++v4;
    goto LABEL_48;
  }
  if ( v7 != 1 )
    return (__int64 *)a2;
LABEL_51:
  if ( *(_BYTE *)*v4 <= 0x1Cu )
    return (__int64 *)a2;
  v36[0] = *v4;
  if ( !sub_2B4B3F0(a3 + 1976, v36) )
    return (__int64 *)a2;
  return v4;
}
