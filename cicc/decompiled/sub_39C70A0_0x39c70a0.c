// Function: sub_39C70A0
// Address: 0x39c70a0
//
__int64 __fastcall sub_39C70A0(__int64 a1, int *a2, size_t a3, __int64 a4)
{
  __int64 v4; // r15
  unsigned int v8; // esi
  __int64 v9; // r8
  unsigned int v10; // ecx
  __int64 *v11; // rax
  __int64 v12; // rdx
  int v14; // r11d
  __int64 *v15; // r10
  int v16; // edi
  int v17; // ecx
  int v18; // eax
  int v19; // esi
  __int64 v20; // rdi
  unsigned int v21; // edx
  __int64 v22; // r8
  int v23; // r10d
  __int64 *v24; // r9
  int v25; // eax
  int v26; // edx
  int v27; // r9d
  __int64 *v28; // r8
  __int64 v29; // rdi
  unsigned int v30; // ebx
  __int64 v31; // rsi
  int v33; // [rsp+10h] [rbp-40h] BYREF
  __int64 v34; // [rsp+18h] [rbp-38h]

  v4 = a1 + 160;
  sub_39C5C80(a1 + 160);
  v8 = *(_DWORD *)(a1 + 184);
  if ( !v8 )
  {
    ++*(_QWORD *)(a1 + 160);
    goto LABEL_16;
  }
  v9 = *(_QWORD *)(a1 + 168);
  v10 = (v8 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
  v11 = (__int64 *)(v9 + 16LL * v10);
  v12 = *v11;
  if ( a4 == *v11 )
    goto LABEL_3;
  v14 = 1;
  v15 = 0;
  while ( v12 != -8 )
  {
    if ( !v15 && v12 == -16 )
      v15 = v11;
    v10 = (v8 - 1) & (v14 + v10);
    v11 = (__int64 *)(v9 + 16LL * v10);
    v12 = *v11;
    if ( a4 == *v11 )
      goto LABEL_3;
    ++v14;
  }
  v16 = *(_DWORD *)(a1 + 176);
  if ( v15 )
    v11 = v15;
  ++*(_QWORD *)(a1 + 160);
  v17 = v16 + 1;
  if ( 4 * (v16 + 1) >= 3 * v8 )
  {
LABEL_16:
    sub_39C5E30(v4, 2 * v8);
    v18 = *(_DWORD *)(a1 + 184);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(a1 + 168);
      v21 = (v18 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
      v17 = *(_DWORD *)(a1 + 176) + 1;
      v11 = (__int64 *)(v20 + 16LL * v21);
      v22 = *v11;
      if ( a4 != *v11 )
      {
        v23 = 1;
        v24 = 0;
        while ( v22 != -8 )
        {
          if ( !v24 && v22 == -16 )
            v24 = v11;
          v21 = v19 & (v23 + v21);
          v11 = (__int64 *)(v20 + 16LL * v21);
          v22 = *v11;
          if ( a4 == *v11 )
            goto LABEL_12;
          ++v23;
        }
        if ( v24 )
          v11 = v24;
      }
      goto LABEL_12;
    }
    goto LABEL_44;
  }
  if ( v8 - *(_DWORD *)(a1 + 180) - v17 <= v8 >> 3 )
  {
    sub_39C5E30(v4, v8);
    v25 = *(_DWORD *)(a1 + 184);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = 1;
      v28 = 0;
      v29 = *(_QWORD *)(a1 + 168);
      v30 = (v25 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
      v17 = *(_DWORD *)(a1 + 176) + 1;
      v11 = (__int64 *)(v29 + 16LL * v30);
      v31 = *v11;
      if ( a4 != *v11 )
      {
        while ( v31 != -8 )
        {
          if ( !v28 && v31 == -16 )
            v28 = v11;
          v30 = v26 & (v27 + v30);
          v11 = (__int64 *)(v29 + 16LL * v30);
          v31 = *v11;
          if ( a4 == *v11 )
            goto LABEL_12;
          ++v27;
        }
        if ( v28 )
          v11 = v28;
      }
      goto LABEL_12;
    }
LABEL_44:
    ++*(_DWORD *)(a1 + 176);
    BUG();
  }
LABEL_12:
  *(_DWORD *)(a1 + 176) = v17;
  if ( *v11 != -8 )
    --*(_DWORD *)(a1 + 180);
  *v11 = a4;
  *((_DWORD *)v11 + 2) = 0;
LABEL_3:
  *((_DWORD *)v11 + 2) = 1;
  if ( a3 )
    sub_16C1A90((int *)a1, a2, a3);
  sub_39C6FD0((int *)a1, a4);
  sub_16C1AA0((_DWORD *)a1, &v33);
  return v34;
}
