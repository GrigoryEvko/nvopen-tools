// Function: sub_252E6C0
// Address: 0x252e6c0
//
__int64 __fastcall sub_252E6C0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned int v4; // r13d
  __int64 v6; // rax
  unsigned __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rdx
  _QWORD *v10; // rdi
  _QWORD *v11; // rsi
  _BYTE *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  _QWORD *v15; // rdi
  _QWORD *v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rsi
  int v20; // edx
  int v21; // ecx
  int v22; // r8d
  unsigned int v23; // edx
  __int64 v24; // rdi
  __int64 v25; // rsi
  int v26; // edx
  int v27; // ecx
  int v28; // r8d
  unsigned int v29; // edx
  __int64 v30; // rdi
  __int64 v31; // [rsp-38h] [rbp-38h] BYREF
  __int64 v32; // [rsp-30h] [rbp-30h] BYREF

  result = (*(_DWORD *)(a2 + 96) >> 3) & 1;
  LOBYTE(result) = *(_DWORD *)(a2 + 96) == 17 || (*(_DWORD *)(a2 + 96) & 8) != 0;
  if ( !(_BYTE)result )
    return 1;
  v4 = *(unsigned __int8 *)(a2 + 24);
  if ( (_BYTE)v4 )
  {
    if ( **(_QWORD **)a1 )
    {
      v6 = *(_QWORD *)(a2 + 8);
      if ( *(_BYTE *)v6 != 85 )
        return 0;
      v18 = *(_QWORD *)(v6 - 32);
      if ( !v18 )
        return 0;
      if ( *(_BYTE *)v18 )
        return 0;
      if ( *(_QWORD *)(v18 + 24) != *(_QWORD *)(v6 + 80) )
        return 0;
      if ( (*(_BYTE *)(v18 + 33) & 0x20) == 0 )
        return 0;
      if ( *(_DWORD *)(v18 + 36) != 11 )
        return 0;
      v7 = *(_QWORD *)(a2 + 16);
      if ( !v7 )
        return 0;
    }
    else
    {
      v7 = *(_QWORD *)(a2 + 16);
      if ( !v7 )
        goto LABEL_12;
    }
    v8 = sub_250C3F0(v7, *(_QWORD *)(**(_QWORD **)(a1 + 8) + 8LL));
    v31 = v8;
    if ( v8 )
    {
      v9 = *(_QWORD *)(a1 + 16);
      if ( *(_DWORD *)(v9 + 16) )
      {
        v19 = *(_QWORD *)(v9 + 8);
        v20 = *(_DWORD *)(v9 + 24);
        if ( v20 )
        {
          v21 = v20 - 1;
          v22 = 1;
          v23 = (v20 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
          v24 = *(_QWORD *)(v19 + 8LL * v23);
          if ( v8 == v24 )
            goto LABEL_16;
          while ( v24 != -4096 )
          {
            v23 = v21 & (v22 + v23);
            v24 = *(_QWORD *)(v19 + 8LL * v23);
            if ( v8 == v24 )
              goto LABEL_16;
            ++v22;
          }
        }
      }
      else
      {
        v10 = *(_QWORD **)(v9 + 32);
        v11 = &v10[*(unsigned int *)(v9 + 40)];
        if ( v11 != sub_25065C0(v10, (__int64)v11, &v31) )
          goto LABEL_16;
      }
    }
LABEL_12:
    v12 = *(_BYTE **)(a2 + 8);
    if ( *v12 != 62 )
      return 0;
    v13 = sub_250C3F0(*((_QWORD *)v12 - 8), *(_QWORD *)(**(_QWORD **)(a1 + 8) + 8LL));
    v31 = v13;
    if ( !v13 )
      return 0;
    v14 = *(_QWORD *)(a1 + 16);
    if ( !*(_DWORD *)(v14 + 16) )
    {
      v15 = *(_QWORD **)(v14 + 32);
      v16 = &v15[*(unsigned int *)(v14 + 40)];
      if ( v16 != sub_25065C0(v15, (__int64)v16, &v31) )
        goto LABEL_16;
      return 0;
    }
    v25 = *(_QWORD *)(v14 + 8);
    v26 = *(_DWORD *)(v14 + 24);
    if ( !v26 )
      return 0;
    v27 = v26 - 1;
    v28 = 1;
    v29 = (v26 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
    v30 = *(_QWORD *)(v25 + 8LL * v29);
    if ( v13 != v30 )
    {
      while ( v30 != -4096 )
      {
        v29 = v27 & (v28 + v29);
        v30 = *(_QWORD *)(v25 + 8LL * v29);
        if ( v13 == v30 )
          goto LABEL_16;
        ++v28;
      }
      return 0;
    }
LABEL_16:
    v17 = *(_QWORD *)(a1 + 24);
    v32 = *(_QWORD *)(a2 + 8);
    sub_252E280(v17, &v32);
    return v4;
  }
  return result;
}
