// Function: sub_16D19C0
// Address: 0x16d19c0
//
__int64 __fastcall sub_16D19C0(__int64 a1, unsigned __int8 *a2, size_t a3)
{
  unsigned int v6; // edi
  unsigned __int8 *v7; // rsi
  unsigned int v8; // r14d
  unsigned __int8 *v9; // rax
  unsigned int v10; // ebx
  int v11; // ecx
  unsigned int v12; // r8d
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  int v16; // r10d
  int v17; // r11d
  _QWORD *v18; // rdx
  __int64 v19; // r9
  unsigned int v21; // r8d
  int v22; // eax
  unsigned int v23; // [rsp+4h] [rbp-4Ch]
  __int64 v24; // [rsp+8h] [rbp-48h]
  int v25; // [rsp+10h] [rbp-40h]
  int v26; // [rsp+14h] [rbp-3Ch]
  __int64 v27; // [rsp+18h] [rbp-38h]

  v6 = *(_DWORD *)(a1 + 8);
  if ( v6 )
  {
    v7 = &a2[a3];
    v8 = v6 - 1;
    if ( &a2[a3] != a2 )
      goto LABEL_3;
  }
  else
  {
    sub_16D1890(a1, 0x10u);
    v6 = *(_DWORD *)(a1 + 8);
    v7 = &a2[a3];
    v8 = v6 - 1;
    if ( &a2[a3] != a2 )
    {
LABEL_3:
      v9 = a2;
      v10 = 0;
      do
      {
        v11 = *v9++;
        v10 += v11 + 32 * v10;
      }
      while ( v7 != v9 );
      v12 = v8 & v10;
      v13 = v8 & v10;
      v14 = 8 * v13;
      goto LABEL_6;
    }
  }
  v12 = 0;
  v14 = 0;
  v13 = 0;
  v10 = 0;
LABEL_6:
  v15 = *(_QWORD *)a1;
  v16 = -1;
  v17 = 1;
  v18 = *(_QWORD **)(*(_QWORD *)a1 + v14);
  v19 = *(_QWORD *)a1 + 8LL * v6 + 8;
  if ( !v18 )
    goto LABEL_7;
  do
  {
    if ( v18 == (_QWORD *)-8LL )
    {
      if ( v16 == -1 )
        v16 = v12;
    }
    else if ( *(_DWORD *)(v19 + 4 * v13) == v10 && *v18 == a3 )
    {
      v25 = v17;
      v24 = v19;
      v26 = v16;
      v27 = v15;
      if ( !a3 )
        return v12;
      v23 = v12;
      v22 = memcmp(a2, (char *)v18 + *(unsigned int *)(a1 + 20), a3);
      v12 = v23;
      v15 = v27;
      v16 = v26;
      v19 = v24;
      v17 = v25;
      if ( !v22 )
        return v12;
    }
    v21 = v17 + v12;
    ++v17;
    v12 = v8 & v21;
    v13 = v12;
    v18 = *(_QWORD **)(v15 + 8LL * v12);
  }
  while ( v18 );
  if ( v16 == -1 )
  {
LABEL_7:
    *(_DWORD *)(v19 + 4 * v13) = v10;
  }
  else
  {
    v12 = v16;
    *(_DWORD *)(v19 + 4LL * v16) = v10;
  }
  return v12;
}
