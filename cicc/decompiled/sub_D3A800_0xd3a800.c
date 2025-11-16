// Function: sub_D3A800
// Address: 0xd3a800
//
__int64 __fastcall sub_D3A800(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v2; // r13
  __int64 v4; // r12
  unsigned int v5; // esi
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned __int64 *v8; // rdi
  int v9; // r11d
  unsigned int v10; // ecx
  unsigned __int64 *v11; // rax
  unsigned __int64 v12; // rdx
  _BYTE *v13; // rsi
  _QWORD *v14; // rdi
  __int64 v15; // r12
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 result; // rax
  int v19; // eax
  int v20; // edx
  int v21; // eax
  int v22; // ecx
  __int64 v23; // r8
  unsigned int v24; // eax
  unsigned __int64 v25; // rsi
  int v26; // r10d
  unsigned __int64 *v27; // r9
  int v28; // eax
  int v29; // eax
  __int64 v30; // rsi
  unsigned __int64 *v31; // r8
  unsigned int v32; // r14d
  int v33; // r9d
  unsigned __int64 v34; // rcx

  v2 = a2 & 0xFFFFFFFFFFFFFFFBLL;
  v4 = *a1;
  v5 = *(_DWORD *)(*a1 + 48LL);
  v6 = *a1 + 24LL;
  if ( !v5 )
  {
    ++*(_QWORD *)(v4 + 24);
    goto LABEL_25;
  }
  v7 = *(_QWORD *)(v4 + 32);
  v8 = 0;
  v9 = 1;
  v10 = (v5 - 1) & (v2 ^ (v2 >> 9));
  v11 = (unsigned __int64 *)(v7 + 32LL * v10);
  v12 = *v11;
  if ( v2 != *v11 )
  {
    while ( v12 != -4 )
    {
      if ( v12 == -16 && !v8 )
        v8 = v11;
      v10 = (v5 - 1) & (v9 + v10);
      v11 = (unsigned __int64 *)(v7 + 32LL * v10);
      v12 = *v11;
      if ( v2 == *v11 )
        goto LABEL_3;
      ++v9;
    }
    if ( !v8 )
      v8 = v11;
    v19 = *(_DWORD *)(v4 + 40);
    ++*(_QWORD *)(v4 + 24);
    v20 = v19 + 1;
    if ( 4 * (v19 + 1) < 3 * v5 )
    {
      if ( v5 - *(_DWORD *)(v4 + 44) - v20 > v5 >> 3 )
      {
LABEL_20:
        *(_DWORD *)(v4 + 40) = v20;
        if ( *v8 != -4 )
          --*(_DWORD *)(v4 + 44);
        *v8 = v2;
        v13 = 0;
        v14 = v8 + 1;
        *v14 = 0;
        v14[1] = 0;
        v14[2] = 0;
        v4 = *a1;
        goto LABEL_23;
      }
      sub_D3A5D0(v6, v5);
      v28 = *(_DWORD *)(v4 + 48);
      if ( v28 )
      {
        v29 = v28 - 1;
        v30 = *(_QWORD *)(v4 + 32);
        v31 = 0;
        v32 = v29 & (v2 ^ (v2 >> 9));
        v33 = 1;
        v20 = *(_DWORD *)(v4 + 40) + 1;
        v8 = (unsigned __int64 *)(v30 + 32LL * v32);
        v34 = *v8;
        if ( v2 != *v8 )
        {
          while ( v34 != -4 )
          {
            if ( !v31 && v34 == -16 )
              v31 = v8;
            v32 = v29 & (v33 + v32);
            v8 = (unsigned __int64 *)(v30 + 32LL * v32);
            v34 = *v8;
            if ( v2 == *v8 )
              goto LABEL_20;
            ++v33;
          }
          if ( v31 )
            v8 = v31;
        }
        goto LABEL_20;
      }
LABEL_48:
      ++*(_DWORD *)(v4 + 40);
      BUG();
    }
LABEL_25:
    sub_D3A5D0(v6, 2 * v5);
    v21 = *(_DWORD *)(v4 + 48);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(v4 + 32);
      v20 = *(_DWORD *)(v4 + 40) + 1;
      v24 = (v21 - 1) & (v2 ^ (v2 >> 9));
      v8 = (unsigned __int64 *)(v23 + 32LL * v24);
      v25 = *v8;
      if ( v2 != *v8 )
      {
        v26 = 1;
        v27 = 0;
        while ( v25 != -4 )
        {
          if ( !v27 && v25 == -16 )
            v27 = v8;
          v24 = v22 & (v26 + v24);
          v8 = (unsigned __int64 *)(v23 + 32LL * v24);
          v25 = *v8;
          if ( v2 == *v8 )
            goto LABEL_20;
          ++v26;
        }
        if ( v27 )
          v8 = v27;
      }
      goto LABEL_20;
    }
    goto LABEL_48;
  }
LABEL_3:
  v13 = (_BYTE *)v11[2];
  v14 = v11 + 1;
  if ( (_BYTE *)v11[3] == v13 )
  {
LABEL_23:
    sub_B8BBF0((__int64)v14, v13, (_DWORD *)(v4 + 200));
    goto LABEL_7;
  }
  if ( v13 )
  {
    *(_DWORD *)v13 = *(_DWORD *)(v4 + 200);
    v13 = (_BYTE *)v11[2];
  }
  v11[2] = (unsigned __int64)(v13 + 4);
LABEL_7:
  v15 = *a1;
  v16 = a1[1];
  v17 = *(unsigned int *)(*a1 + 64LL);
  if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(*a1 + 68LL) )
  {
    sub_C8D5F0(v15 + 56, (const void *)(v15 + 72), v17 + 1, 8u, v6, v7);
    v17 = *(unsigned int *)(v15 + 64);
  }
  *(_QWORD *)(*(_QWORD *)(v15 + 56) + 8 * v17) = v16;
  ++*(_DWORD *)(v15 + 64);
  result = *a1;
  ++*(_DWORD *)(*a1 + 200LL);
  return result;
}
