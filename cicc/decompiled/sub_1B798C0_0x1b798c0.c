// Function: sub_1B798C0
// Address: 0x1b798c0
//
__int64 __fastcall sub_1B798C0(__int64 a1, __int64 a2, _QWORD *a3)
{
  int v4; // r15d
  __int64 v6; // rbx
  unsigned __int64 v7; // rax
  int v8; // ecx
  const void *v9; // rdi
  __int64 v10; // r8
  int v11; // r14d
  int v12; // r9d
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rax
  size_t v17; // rdx
  unsigned int i; // r15d
  __int64 v19; // r12
  const void *v20; // rsi
  unsigned int v21; // r15d
  int v22; // eax
  __int64 v23; // [rsp+0h] [rbp-50h]
  int v24; // [rsp+8h] [rbp-48h]
  int v25; // [rsp+Ch] [rbp-44h]
  size_t v26; // [rsp+10h] [rbp-40h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = sub_16D3930(*(_QWORD **)a2, *(_QWORD *)(a2 + 8));
  v8 = v4 - 1;
  v9 = *(const void **)a2;
  v10 = 0;
  v11 = *(_DWORD *)(a2 + 16);
  v12 = 1;
  v13 = ((unsigned int)(37 * v11) | (v7 << 32)) - 1 - ((unsigned __int64)(unsigned int)(37 * v11) << 32);
  v14 = ((v13 >> 22) ^ v13) - 1 - (((v13 >> 22) ^ v13) << 13);
  v15 = ((9 * ((v14 >> 8) ^ v14)) >> 15) ^ (9 * ((v14 >> 8) ^ v14));
  v16 = ((v15 - 1 - (v15 << 27)) >> 31) ^ (v15 - 1 - (v15 << 27));
  v17 = *(_QWORD *)(a2 + 8);
  for ( i = (v4 - 1) & v16; ; i = v8 & v21 )
  {
    v19 = v6 + 56LL * i;
    v20 = *(const void **)v19;
    if ( *(_QWORD *)v19 != -1 )
    {
      if ( v20 == (const void *)-2LL )
      {
        if ( v9 == (const void *)-2LL && v11 == *(_DWORD *)(v19 + 16) )
          goto LABEL_17;
        if ( *(_DWORD *)(v19 + 16) == -2 && !v10 )
          v10 = v6 + 56LL * i;
      }
      else if ( v17 == *(_QWORD *)(v19 + 8) )
      {
        if ( !v17 )
          goto LABEL_16;
        v24 = v12;
        v23 = v10;
        v25 = v8;
        v26 = v17;
        v22 = memcmp(v9, v20, v17);
        v17 = v26;
        v8 = v25;
        v10 = v23;
        v12 = v24;
        if ( !v22 )
        {
LABEL_16:
          if ( v11 == *(_DWORD *)(v19 + 16) )
          {
LABEL_17:
            *a3 = v19;
            return 1;
          }
        }
      }
      goto LABEL_7;
    }
    if ( v9 == (const void *)-1LL && v11 == *(_DWORD *)(v19 + 16) )
      goto LABEL_17;
    if ( *(_DWORD *)(v19 + 16) == -1 )
      break;
LABEL_7:
    v21 = v12 + i;
    ++v12;
  }
  if ( !v10 )
    v10 = v6 + 56LL * i;
  *a3 = v10;
  return 0;
}
