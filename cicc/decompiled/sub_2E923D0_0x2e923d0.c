// Function: sub_2E923D0
// Address: 0x2e923d0
//
unsigned __int64 __fastcall sub_2E923D0(unsigned __int64 a1, int a2, __int64 a3)
{
  unsigned __int64 i; // rbx
  __int64 v6; // r13
  __int64 v7; // r14
  __int64 v8; // r12
  char v9; // r11
  char v10; // r10
  bool v11; // al
  __int64 v12; // r9
  unsigned __int64 v13; // r8
  __int64 v14; // rdx
  __int64 *v15; // rdx
  char v16; // dl
  __int64 v17; // r14
  __int64 v18; // rdx
  unsigned __int64 v19; // rdx
  bool v20; // zf
  char v22; // [rsp+5h] [rbp-6Bh]
  bool v23; // [rsp+6h] [rbp-6Ah]
  char v24; // [rsp+7h] [rbp-69h]
  __int64 v25; // [rsp+8h] [rbp-68h]
  unsigned __int64 v26; // [rsp+10h] [rbp-60h]
  const void *v27; // [rsp+20h] [rbp-50h]
  unsigned __int64 v28; // [rsp+28h] [rbp-48h]
  unsigned __int16 v29; // [rsp+3Dh] [rbp-33h]

  for ( i = a1; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
    ;
  v6 = *(_QWORD *)(a1 + 24) + 48LL;
  do
  {
    v7 = *(_QWORD *)(i + 32);
    v8 = v7 + 40LL * (*(_DWORD *)(i + 40) & 0xFFFFFF);
    if ( v7 != v8 )
      goto LABEL_7;
    i = *(_QWORD *)(i + 8);
  }
  while ( v6 != i && (*(_BYTE *)(i + 44) & 4) != 0 );
  i = *(_QWORD *)(a1 + 24) + 48LL;
  if ( v8 != v7 )
  {
LABEL_7:
    v9 = 0;
    v10 = 0;
    v11 = 0;
    v27 = (const void *)(a3 + 16);
    while ( 1 )
    {
      if ( *(_BYTE *)v7 || *(_DWORD *)(v7 + 8) != a2 )
        goto LABEL_17;
      if ( a3 )
      {
        v12 = *(_QWORD *)(v7 + 16);
        v13 = v28 & 0xFFFFFFFF00000000LL | (-858993459 * (unsigned int)((v7 - *(_QWORD *)(i + 32)) >> 3));
        v14 = *(unsigned int *)(a3 + 8);
        v28 = v13;
        if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
        {
          v22 = v10;
          v23 = v11;
          v24 = v9;
          v25 = *(_QWORD *)(v7 + 16);
          v26 = v13;
          sub_C8D5F0(a3, v27, v14 + 1, 0x10u, v13, v12);
          v10 = v22;
          v11 = v23;
          v9 = v24;
          v14 = *(unsigned int *)(a3 + 8);
          v12 = v25;
          v13 = v26;
        }
        v15 = (__int64 *)(*(_QWORD *)a3 + 16 * v14);
        *v15 = v12;
        v15[1] = v13;
        ++*(_DWORD *)(a3 + 8);
      }
      v16 = *(_BYTE *)(v7 + 3) & 0x10;
      if ( (*(_BYTE *)(v7 + 4) & 1) != 0 || (*(_BYTE *)(v7 + 4) & 2) != 0 )
      {
        if ( v16 )
        {
          v9 = 1;
          goto LABEL_17;
        }
        if ( v11 )
          goto LABEL_17;
      }
      else
      {
        if ( v16 )
        {
          v9 = 1;
          if ( ((*(_DWORD *)v7 >> 8) & 0xFFF) != 0 )
          {
            v11 = 1;
            v10 = 1;
          }
          goto LABEL_17;
        }
        v10 = 1;
        if ( v11 )
          goto LABEL_17;
      }
      v19 = *(_QWORD *)(*(_QWORD *)(v7 + 16) + 32LL)
          + 0xFFFFFFF800000008LL * (unsigned int)((v7 - *(_QWORD *)(i + 32)) >> 3);
      if ( !*(_BYTE *)v19 && (*(_BYTE *)(v19 + 3) & 0x10) == 0 )
      {
        v20 = (*(_WORD *)(v19 + 2) & 0xFF0) == 0;
        v18 = v8;
        v11 = !v20;
        v17 = v7 + 40;
        if ( v17 == v8 )
          goto LABEL_21;
LABEL_31:
        v8 = v17;
        goto LABEL_24;
      }
LABEL_17:
      v17 = v7 + 40;
      v18 = v8;
      if ( v17 != v8 )
        goto LABEL_31;
LABEL_21:
      while ( 1 )
      {
        i = *(_QWORD *)(i + 8);
        if ( v6 == i || (*(_BYTE *)(i + 44) & 4) == 0 )
          break;
        v8 = *(_QWORD *)(i + 32);
        v18 = v8 + 40LL * (*(_DWORD *)(i + 40) & 0xFFFFFF);
        if ( v8 != v18 )
          goto LABEL_24;
      }
      if ( v18 == v8 )
        goto LABEL_34;
      i = v6;
LABEL_24:
      v7 = v8;
      v8 = v18;
    }
  }
  v9 = 0;
  v10 = 0;
  v11 = 0;
LABEL_34:
  LOBYTE(v29) = v10;
  HIBYTE(v29) = v9;
  return v29 | ((unsigned __int64)v11 << 16);
}
