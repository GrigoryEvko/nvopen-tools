// Function: sub_BB0290
// Address: 0xbb0290
//
__int64 __fastcall sub_BB0290(__int64 a1, _QWORD *a2, __int64 a3)
{
  unsigned int v6; // esi
  int v7; // eax
  int v8; // eax
  int v10; // r14d
  int v11; // r15d
  int v12; // r15d
  __int64 v13; // r14
  int v14; // eax
  char *v15; // rdi
  int v16; // r9d
  __int64 v17; // r8
  unsigned int j; // ecx
  bool v19; // al
  const void *v20; // rsi
  size_t v21; // rdx
  unsigned int v22; // ecx
  int v23; // eax
  int v24; // r14d
  __int64 v25; // r15
  int v26; // eax
  const void *v27; // rdi
  int v28; // r9d
  unsigned int i; // ecx
  const void *v30; // rsi
  size_t v31; // rdx
  unsigned int v32; // ecx
  int v33; // eax
  __int64 v34; // [rsp+10h] [rbp-40h]
  __int64 v35; // [rsp+10h] [rbp-40h]
  int v36; // [rsp+18h] [rbp-38h]
  int v37; // [rsp+18h] [rbp-38h]
  unsigned int v38; // [rsp+1Ch] [rbp-34h]
  unsigned int v39; // [rsp+1Ch] [rbp-34h]

  v6 = *(_DWORD *)(a1 + 24);
  v7 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v8 = v7 + 1;
  if ( 4 * v8 >= 3 * v6 )
  {
    a3 = 0;
    sub_A2B260(a1, 2 * v6);
    v10 = *(_DWORD *)(a1 + 24);
    if ( v10 )
    {
      v24 = v10 - 1;
      v25 = *(_QWORD *)(a1 + 8);
      v26 = sub_C94890(*a2, a2[1]);
      v27 = (const void *)*a2;
      v28 = 1;
      v17 = 0;
      for ( i = v24 & v26; ; i = v24 & v32 )
      {
        a3 = v25 + 24LL * i;
        v30 = *(const void **)a3;
        if ( *(_QWORD *)a3 == -1 )
          break;
        if ( v30 == (const void *)-2LL )
        {
          if ( v27 == (const void *)-2LL )
            goto LABEL_7;
          if ( !v17 )
            v17 = v25 + 24LL * i;
        }
        else
        {
          v31 = a2[1];
          if ( v31 == *(_QWORD *)(a3 + 8) )
          {
            v37 = v28;
            v35 = v17;
            v39 = i;
            if ( !v31 )
              goto LABEL_7;
            v33 = memcmp(v27, v30, v31);
            i = v39;
            v17 = v35;
            v28 = v37;
            if ( !v33 )
              goto LABEL_7;
          }
        }
        v32 = v28 + i;
        ++v28;
      }
      if ( v27 != (const void *)-1LL )
      {
LABEL_20:
        if ( v17 )
          a3 = v17;
      }
    }
  }
  else
  {
    if ( v6 - *(_DWORD *)(a1 + 20) - v8 > v6 >> 3 )
      goto LABEL_3;
    sub_A2B260(a1, v6);
    v11 = *(_DWORD *)(a1 + 24);
    a3 = 0;
    if ( v11 )
    {
      v12 = v11 - 1;
      v13 = *(_QWORD *)(a1 + 8);
      v14 = sub_C94890(*a2, a2[1]);
      v15 = (char *)*a2;
      v16 = 1;
      v17 = 0;
      for ( j = v12 & v14; ; j = v12 & v22 )
      {
        a3 = v13 + 24LL * j;
        v19 = v15 + 1 == 0;
        v20 = *(const void **)a3;
        if ( *(_QWORD *)a3 != -1 )
        {
          v19 = v15 + 2 == 0;
          if ( v20 != (const void *)-2LL )
          {
            v21 = a2[1];
            if ( *(_QWORD *)(a3 + 8) != v21 )
              goto LABEL_13;
            v36 = v16;
            v34 = v17;
            v38 = j;
            if ( !v21 )
              break;
            v23 = memcmp(v15, v20, v21);
            j = v38;
            v17 = v34;
            v16 = v36;
            v19 = v23 == 0;
          }
        }
        if ( v19 )
          break;
        if ( v20 == (const void *)-1LL )
          goto LABEL_20;
LABEL_13:
        if ( v17 || v20 != (const void *)-2LL )
          a3 = v17;
        v22 = v16 + j;
        v17 = a3;
        ++v16;
      }
    }
  }
LABEL_7:
  v8 = *(_DWORD *)(a1 + 16) + 1;
LABEL_3:
  *(_DWORD *)(a1 + 16) = v8;
  if ( *(_QWORD *)a3 != -1 )
    --*(_DWORD *)(a1 + 20);
  return a3;
}
