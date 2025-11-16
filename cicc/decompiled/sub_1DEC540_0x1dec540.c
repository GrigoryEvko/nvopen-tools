// Function: sub_1DEC540
// Address: 0x1dec540
//
__int64 __fastcall sub_1DEC540(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v6; // rdi
  char v7; // dl
  __int64 v8; // r8
  __int64 v9; // r9
  int v10; // r11d
  __int64 v11; // rsi
  unsigned int v12; // eax
  __int64 *v13; // rcx
  __int64 v14; // r10
  __int64 v15; // rsi
  char v16; // al
  __int64 v18; // rsi
  unsigned int v19; // eax
  int v20; // edi
  unsigned int v21; // r8d
  __int64 v22; // rsi
  __int64 v23; // rax
  int v24; // r14d
  __int64 *v25; // rsi
  __int64 *v26; // [rsp+8h] [rbp-38h] BYREF

  v6 = *(_QWORD *)a2;
  v7 = *(_BYTE *)(a2 + 8) & 1;
  if ( v7 )
  {
    v8 = *a3;
    v9 = a2 + 16;
    v10 = 15;
    v11 = 128;
    v12 = ((unsigned __int8)((unsigned int)*a3 >> 4) ^ (unsigned __int8)((unsigned int)*a3 >> 9)) & 0xF;
    v13 = (__int64 *)(v9
                    + 8LL
                    * (((unsigned __int8)((unsigned int)*a3 >> 4) ^ (unsigned __int8)((unsigned int)*a3 >> 9)) & 0xF));
    v14 = *v13;
    if ( *v13 == *a3 )
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
    v12 = (v18 - 1) & (((unsigned int)*a3 >> 4) ^ ((unsigned int)*a3 >> 9));
    v13 = (__int64 *)(v9 + 8LL * v12);
    v14 = *v13;
    if ( *v13 == *a3 )
    {
LABEL_7:
      v11 = 8 * v18;
      goto LABEL_3;
    }
LABEL_18:
    v24 = 1;
    v25 = 0;
    while ( 1 )
    {
      if ( v14 == -8 )
      {
        v19 = *(_DWORD *)(a2 + 8);
        if ( v25 )
          v13 = v25;
        *(_QWORD *)a2 = v6 + 1;
        v20 = (v19 >> 1) + 1;
        if ( v7 )
        {
          v21 = 48;
          LODWORD(v18) = 16;
          goto LABEL_10;
        }
        LODWORD(v18) = *(_DWORD *)(a2 + 24);
        goto LABEL_9;
      }
      if ( v14 == -16 && !v25 )
        v25 = v13;
      v12 = v10 & (v24 + v12);
      v13 = (__int64 *)(v9 + 8LL * v12);
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
    v11 = 128;
    goto LABEL_3;
  }
  v19 = *(_DWORD *)(a2 + 8);
  v13 = 0;
  *(_QWORD *)a2 = v6 + 1;
  v20 = (v19 >> 1) + 1;
LABEL_9:
  v21 = 3 * v18;
LABEL_10:
  if ( v21 <= 4 * v20 )
  {
    LODWORD(v18) = 2 * v18;
    goto LABEL_25;
  }
  if ( (int)v18 - *(_DWORD *)(a2 + 12) - v20 <= (unsigned int)v18 >> 3 )
  {
LABEL_25:
    sub_1DEC180(a2, v18);
    sub_1DE9010(a2, a3, &v26);
    v13 = v26;
    v19 = *(_DWORD *)(a2 + 8);
  }
  *(_DWORD *)(a2 + 8) = (2 * (v19 >> 1) + 2) | v19 & 1;
  if ( *v13 != -8 )
    --*(_DWORD *)(a2 + 12);
  *v13 = *a3;
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v22 = a2 + 16;
    v23 = 128;
  }
  else
  {
    v22 = *(_QWORD *)(a2 + 16);
    v23 = 8LL * *(unsigned int *)(a2 + 24);
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
