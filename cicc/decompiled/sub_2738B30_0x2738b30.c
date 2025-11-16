// Function: sub_2738B30
// Address: 0x2738b30
//
void __fastcall sub_2738B30(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v7; // r13
  _QWORD *v10; // rdx
  __int64 v11; // rax
  unsigned __int64 *v12; // rax
  __int64 *v13; // rsi
  __int64 v14; // rax
  _QWORD *v15; // rdx
  __int64 *v16; // r8
  int v17; // edi
  __int64 v18; // r9
  __int64 v19; // r10
  int v20; // edi
  unsigned int v21; // ecx
  __int64 *v22; // rax
  __int64 v23; // r11
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26; // rdi
  int v27; // eax
  int v28; // [rsp+Ch] [rbp-34h]

  v7 = a2;
  v10 = a2 + 79;
  if ( !*(_BYTE *)(a1 + 8) )
    v10 = a2;
  v11 = (unsigned int)(*((_DWORD *)v10 + 4) - 1);
  *((_DWORD *)v10 + 4) = v11;
  v12 = (unsigned __int64 *)(v10[1] + 144 * v11);
  if ( (unsigned __int64 *)*v12 != v12 + 2 )
    _libc_free(*v12);
  if ( *(_BYTE *)(a1 + 8) )
  {
    v13 = *(__int64 **)(a1 + 16);
    v14 = *(unsigned int *)(a1 + 24);
    v15 = v7 + 154;
    v16 = &v13[v14];
    if ( v13 == v16 )
    {
LABEL_17:
      v7 += 79;
      goto LABEL_12;
    }
  }
  else
  {
    v13 = *(__int64 **)(a1 + 16);
    v14 = *(unsigned int *)(a1 + 24);
    v15 = v7 + 75;
    v16 = &v13[v14];
    if ( v16 == v13 )
      goto LABEL_12;
  }
  do
  {
    v17 = *((_DWORD *)v15 + 6);
    v18 = *v13;
    v19 = v15[1];
    if ( v17 )
    {
      v20 = v17 - 1;
      v21 = v20 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v22 = (__int64 *)(v19 + 16LL * v21);
      v23 = *v22;
      if ( v18 == *v22 )
      {
LABEL_9:
        *v22 = -8192;
        --*((_DWORD *)v15 + 4);
        ++*((_DWORD *)v15 + 5);
      }
      else
      {
        v27 = 1;
        while ( v23 != -4096 )
        {
          v21 = v20 & (v27 + v21);
          v28 = v27 + 1;
          v22 = (__int64 *)(v19 + 16LL * v21);
          v23 = *v22;
          if ( v18 == *v22 )
            goto LABEL_9;
          v27 = v28;
        }
      }
    }
    ++v13;
  }
  while ( v16 != v13 );
  v14 = *(unsigned int *)(a1 + 24);
  if ( *(_BYTE *)(a1 + 8) )
    goto LABEL_17;
LABEL_12:
  *v7 -= v14;
  v24 = (unsigned int)(*(_DWORD *)(a5 + 8) - 1);
  *(_DWORD *)(a5 + 8) = v24;
  v25 = *(_QWORD *)a5 + 48 * v24;
  v26 = *(_QWORD *)(v25 + 16);
  if ( v26 != v25 + 32 )
    _libc_free(v26);
  if ( a3 )
    --*(_DWORD *)(a4 + 8);
}
