// Function: sub_253BC70
// Address: 0x253bc70
//
char __fastcall sub_253BC70(__int64 a1, _QWORD *a2, size_t a3)
{
  char result; // al
  __int64 v6; // r13
  __int64 v7; // r12
  int v8; // r13d
  int v9; // eax
  __int64 v10; // r8
  int v11; // r10d
  unsigned int i; // r9d
  __int64 v13; // rcx
  const void *v14; // rsi
  bool v15; // al
  unsigned int v16; // r9d
  int v17; // eax
  __int64 v18; // r13
  __int64 v19; // r12
  int v20; // eax
  int v21; // r8d
  __int64 v22; // rcx
  int v23; // r10d
  unsigned int j; // r9d
  __int64 v25; // r13
  const void *v26; // rsi
  unsigned int v27; // r9d
  int v28; // eax
  __int64 v29; // [rsp+0h] [rbp-50h]
  int v30; // [rsp+8h] [rbp-48h]
  int v31; // [rsp+10h] [rbp-40h]
  unsigned int v32; // [rsp+10h] [rbp-40h]
  unsigned int v33; // [rsp+14h] [rbp-3Ch]
  int v34; // [rsp+14h] [rbp-3Ch]
  __int64 v35; // [rsp+18h] [rbp-38h]
  __int64 v36; // [rsp+18h] [rbp-38h]
  __int64 v37; // [rsp+18h] [rbp-38h]
  __int64 v38; // [rsp+18h] [rbp-38h]

  if ( !*(_DWORD *)(a1 + 160) )
  {
    result = *(_BYTE *)(a1 + 136);
    if ( !result )
      return result;
  }
  v6 = *(unsigned int *)(a1 + 168);
  v35 = *(_QWORD *)(a1 + 152);
  v7 = v35 + 16 * v6;
  if ( !(_DWORD)v6 )
    goto LABEL_18;
  v8 = v6 - 1;
  v9 = sub_C94890(a2, a3);
  v10 = v35;
  v11 = 1;
  for ( i = v8 & v9; ; i = v8 & v16 )
  {
    v13 = v10 + 16LL * i;
    v14 = *(const void **)v13;
    if ( *(_QWORD *)v13 == -1 )
      break;
    v15 = (_QWORD *)((char *)a2 + 2) == 0;
    if ( v14 != (const void *)-2LL )
    {
      if ( a3 != *(_QWORD *)(v13 + 8) )
        goto LABEL_8;
      v31 = v11;
      v33 = i;
      v36 = v10;
      if ( !a3 )
        goto LABEL_14;
      v29 = v10 + 16LL * i;
      v17 = memcmp(a2, v14, a3);
      v13 = v29;
      v10 = v36;
      i = v33;
      v11 = v31;
      v15 = v17 == 0;
    }
    if ( v15 )
      goto LABEL_14;
    if ( v14 == (const void *)-1LL )
      goto LABEL_13;
LABEL_8:
    v16 = v11 + i;
    ++v11;
  }
  if ( a2 != (_QWORD *)-1LL )
LABEL_13:
    v13 = *(_QWORD *)(a1 + 152) + 16LL * *(unsigned int *)(a1 + 168);
LABEL_14:
  result = 1;
  if ( v7 == v13 )
  {
LABEL_18:
    result = 0;
    v18 = *(unsigned int *)(a1 + 128);
    v37 = *(_QWORD *)(a1 + 112);
    v19 = v37 + 16 * v18;
    if ( (_DWORD)v18 )
    {
      v20 = sub_C94890(a2, a3);
      v21 = v18 - 1;
      v22 = v37;
      v23 = 1;
      for ( j = (v18 - 1) & v20; ; j = v21 & v27 )
      {
        v25 = v22 + 16LL * j;
        v26 = *(const void **)v25;
        if ( *(_QWORD *)v25 == -1 )
          break;
        if ( v26 == (const void *)-2LL )
        {
          if ( a2 == (_QWORD *)-2LL )
            return v19 != v25;
        }
        else if ( a3 == *(_QWORD *)(v25 + 8) )
        {
          v30 = v23;
          v32 = j;
          v34 = v21;
          v38 = v22;
          if ( !a3 )
            return v19 != v25;
          v28 = memcmp(a2, v26, a3);
          v22 = v38;
          v21 = v34;
          j = v32;
          v23 = v30;
          if ( !v28 )
            return v19 != v25;
        }
        v27 = v23 + j;
        ++v23;
      }
      if ( a2 == (_QWORD *)-1LL )
        return v19 != v25;
      return v19 != *(_QWORD *)(a1 + 112) + 16LL * *(unsigned int *)(a1 + 128);
    }
  }
  return result;
}
