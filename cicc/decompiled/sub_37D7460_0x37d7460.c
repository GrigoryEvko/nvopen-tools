// Function: sub_37D7460
// Address: 0x37d7460
//
__int64 __fastcall sub_37D7460(__int64 a1, __int64 a2, int *a3)
{
  __int64 v6; // rcx
  char v7; // al
  int v8; // edi
  __int64 v9; // r8
  int v10; // r11d
  __int64 v11; // rsi
  unsigned int v12; // r9d
  int *v13; // rdx
  int v14; // r10d
  __int64 v15; // rsi
  char v16; // al
  __int64 v18; // rsi
  unsigned int v19; // edx
  int v20; // ecx
  unsigned int v21; // edi
  __int64 v22; // rsi
  __int64 v23; // rax
  int v24; // r14d
  int *v25; // rsi
  int *v26; // [rsp+8h] [rbp-38h] BYREF

  v6 = *(_QWORD *)a2;
  v7 = *(_BYTE *)(a2 + 8) & 1;
  if ( v7 )
  {
    v8 = *a3;
    v9 = a2 + 16;
    v10 = 3;
    v11 = 16;
    v12 = *a3 & 3;
    v13 = (int *)(v9 + 4LL * v12);
    v14 = *v13;
    if ( v8 == *v13 )
    {
LABEL_3:
      v15 = v9 + v11;
      v16 = 0;
      goto LABEL_4;
    }
    goto LABEL_18;
  }
  v18 = *(unsigned int *)(a2 + 24);
  if ( (_DWORD)v18 )
  {
    v8 = *a3;
    v10 = v18 - 1;
    v9 = *(_QWORD *)(a2 + 16);
    v12 = (v18 - 1) & (37 * *a3);
    v13 = (int *)(v9 + 4LL * v12);
    v14 = *v13;
    if ( v8 == *v13 )
    {
LABEL_7:
      v11 = 4 * v18;
      goto LABEL_3;
    }
LABEL_18:
    v24 = 1;
    v25 = 0;
    while ( 1 )
    {
      if ( v14 == -1 )
      {
        if ( !v25 )
          v25 = v13;
        v19 = *(_DWORD *)(a2 + 8);
        *(_QWORD *)a2 = v6 + 1;
        v26 = v25;
        v20 = (v19 >> 1) + 1;
        if ( v7 )
        {
          v21 = 12;
          LODWORD(v18) = 4;
          goto LABEL_10;
        }
        LODWORD(v18) = *(_DWORD *)(a2 + 24);
        goto LABEL_9;
      }
      if ( v14 == -2 && !v25 )
        v25 = v13;
      v12 = v10 & (v24 + v12);
      v13 = (int *)(v9 + 4LL * v12);
      v14 = *v13;
      if ( *v13 == v8 )
        break;
      ++v24;
    }
    if ( !v7 )
    {
      v18 = *(unsigned int *)(a2 + 24);
      goto LABEL_7;
    }
    v11 = 16;
    goto LABEL_3;
  }
  v19 = *(_DWORD *)(a2 + 8);
  v26 = 0;
  *(_QWORD *)a2 = v6 + 1;
  v20 = (v19 >> 1) + 1;
LABEL_9:
  v21 = 3 * v18;
LABEL_10:
  if ( 4 * v20 >= v21 )
  {
    LODWORD(v18) = 2 * v18;
    goto LABEL_25;
  }
  if ( (int)v18 - *(_DWORD *)(a2 + 12) - v20 <= (unsigned int)v18 >> 3 )
  {
LABEL_25:
    sub_B47550(a2, v18);
    sub_37C0030(a2, a3, &v26);
    v19 = *(_DWORD *)(a2 + 8);
  }
  *(_DWORD *)(a2 + 8) = (2 * (v19 >> 1) + 2) | v19 & 1;
  v13 = v26;
  if ( *v26 != -1 )
    --*(_DWORD *)(a2 + 12);
  *v13 = *a3;
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v22 = a2 + 16;
    v23 = 16;
  }
  else
  {
    v22 = *(_QWORD *)(a2 + 16);
    v23 = 4LL * *(unsigned int *)(a2 + 24);
  }
  v15 = v23 + v22;
  v6 = *(_QWORD *)a2;
  v16 = 1;
LABEL_4:
  *(_QWORD *)a1 = a2;
  *(_BYTE *)(a1 + 32) = v16;
  *(_QWORD *)(a1 + 8) = v6;
  *(_QWORD *)(a1 + 16) = v13;
  *(_QWORD *)(a1 + 24) = v15;
  return a1;
}
