// Function: sub_22ECC70
// Address: 0x22ecc70
//
__int64 __fastcall sub_22ECC70(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // rdx
  __int64 v6; // r12
  __int64 v7; // rdi
  __int64 v8; // rsi
  int v9; // eax
  int v10; // ecx
  unsigned int v11; // eax
  __int64 v12; // r8
  int v14; // r9d
  int v15; // ecx
  __int64 v16; // rsi
  int v17; // ecx
  int v18; // edi
  __int64 v19; // r12
  unsigned int v20; // eax
  __int64 v21; // rdx

  v3 = *(_QWORD *)(a2 + 16);
  if ( !v3 )
    return 0;
  while ( 1 )
  {
    v4 = *(_QWORD *)(v3 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v4 - 30) <= 0xAu )
      break;
    v3 = *(_QWORD *)(v3 + 8);
    if ( !v3 )
      return 0;
  }
LABEL_5:
  v5 = (unsigned int)sub_BD2910(v3);
  if ( (*(_BYTE *)(v4 + 7) & 0x40) != 0 )
    v6 = *(_QWORD *)(v4 - 8);
  else
    v6 = v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF);
  v7 = *(_QWORD *)(a1 + 16);
  v8 = *(_QWORD *)(*(_QWORD *)(v3 + 24) + 40LL);
  v9 = *(_DWORD *)(a1 + 32);
  if ( v9 )
  {
    v10 = v9 - 1;
    v11 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v12 = *(_QWORD *)(v7 + 8LL * v11);
    if ( v8 == v12 )
      goto LABEL_9;
    v14 = 1;
    while ( v12 != -4096 )
    {
      v11 = v10 & (v14 + v11);
      v12 = *(_QWORD *)(v7 + 8LL * v11);
      if ( v8 == v12 )
        goto LABEL_9;
      ++v14;
    }
  }
  v15 = *(_DWORD *)(a1 + 80);
  v16 = *(_QWORD *)(a1 + 64);
  if ( v15 )
  {
    v17 = v15 - 1;
    v18 = 1;
    v19 = 32 * v5 + v6;
    v20 = v17 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
    v21 = *(_QWORD *)(v16 + 8LL * v20);
    if ( v19 == v21 )
      goto LABEL_9;
    while ( v21 != -4096 )
    {
      v20 = v17 & (v18 + v20);
      v21 = *(_QWORD *)(v16 + 8LL * v20);
      if ( v19 == v21 )
      {
LABEL_9:
        while ( 1 )
        {
          v3 = *(_QWORD *)(v3 + 8);
          if ( !v3 )
            return 0;
          v4 = *(_QWORD *)(v3 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v4 - 30) <= 0xAu )
            goto LABEL_5;
        }
      }
      ++v18;
    }
  }
  return 1;
}
