// Function: sub_1B57D30
// Address: 0x1b57d30
//
__int64 __fastcall sub_1B57D30(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4)
{
  __int64 v8; // rdi
  char v9; // cl
  __int64 v10; // r8
  __int64 v11; // rsi
  __int64 v12; // r15
  int v13; // r11d
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // r10
  __int64 v18; // r15
  unsigned int v19; // eax
  __int64 *v20; // r9
  int v21; // edx
  unsigned int v22; // esi
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  int v26; // r15d
  unsigned int v27; // esi
  __int64 *v28; // [rsp+18h] [rbp-38h] BYREF

  v8 = *(_QWORD *)a2;
  v9 = *(_BYTE *)(a2 + 8) & 1;
  if ( v9 )
  {
    v10 = a2 + 16;
    v11 = *a3;
    v12 = 64;
    v13 = 3;
    v14 = ((unsigned __int8)((unsigned int)*a3 >> 4) ^ (unsigned __int8)((unsigned int)*a3 >> 9)) & 3;
    v15 = (__int64 *)(v10 + 16LL * v14);
    v16 = *v15;
    if ( v11 == *v15 )
    {
LABEL_3:
      *(_QWORD *)a1 = a2;
      *(_QWORD *)(a1 + 8) = v8;
      *(_QWORD *)(a1 + 16) = v15;
      *(_QWORD *)(a1 + 24) = v10 + v12;
      *(_BYTE *)(a1 + 32) = 0;
      return a1;
    }
    goto LABEL_18;
  }
  v18 = *(unsigned int *)(a2 + 24);
  if ( (_DWORD)v18 )
  {
    v10 = *(_QWORD *)(a2 + 16);
    v11 = *a3;
    v13 = v18 - 1;
    v14 = (v18 - 1) & (((unsigned int)*a3 >> 4) ^ ((unsigned int)*a3 >> 9));
    v15 = (__int64 *)(v10 + 16LL * v14);
    v16 = *v15;
    if ( v11 == *v15 )
    {
LABEL_7:
      v12 = 16 * v18;
      goto LABEL_3;
    }
LABEL_18:
    v26 = 1;
    v20 = 0;
    while ( 1 )
    {
      if ( v16 == -8 )
      {
        if ( !v20 )
          v20 = v15;
        v19 = *(_DWORD *)(a2 + 8);
        *(_QWORD *)a2 = v8 + 1;
        v21 = (v19 >> 1) + 1;
        if ( v9 )
        {
          v22 = 12;
          LODWORD(v18) = 4;
          goto LABEL_10;
        }
        LODWORD(v18) = *(_DWORD *)(a2 + 24);
        goto LABEL_9;
      }
      if ( !v20 && v16 == -16 )
        v20 = v15;
      v14 = v13 & (v26 + v14);
      v15 = (__int64 *)(v10 + 16LL * v14);
      v16 = *v15;
      if ( *v15 == v11 )
        break;
      ++v26;
    }
    if ( !v9 )
    {
      v18 = *(unsigned int *)(a2 + 24);
      goto LABEL_7;
    }
    v12 = 64;
    goto LABEL_3;
  }
  v19 = *(_DWORD *)(a2 + 8);
  v20 = 0;
  *(_QWORD *)a2 = v8 + 1;
  v21 = (v19 >> 1) + 1;
LABEL_9:
  v22 = 3 * v18;
LABEL_10:
  if ( v22 <= 4 * v21 )
  {
    v27 = 2 * v18;
LABEL_25:
    sub_1B57950(a2, v27);
    sub_1B506B0(a2, a3, &v28);
    v20 = v28;
    v19 = *(_DWORD *)(a2 + 8);
    goto LABEL_12;
  }
  if ( (int)v18 - *(_DWORD *)(a2 + 12) - v21 <= (unsigned int)v18 >> 3 )
  {
    v27 = v18;
    goto LABEL_25;
  }
LABEL_12:
  *(_DWORD *)(a2 + 8) = (2 * (v19 >> 1) + 2) | v19 & 1;
  if ( *v20 != -8 )
    --*(_DWORD *)(a2 + 12);
  *v20 = *a3;
  v20[1] = *a4;
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v23 = a2 + 16;
    v24 = 64;
  }
  else
  {
    v23 = *(_QWORD *)(a2 + 16);
    v24 = 16LL * *(unsigned int *)(a2 + 24);
  }
  v25 = *(_QWORD *)a2;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = v20;
  *(_QWORD *)(a1 + 8) = v25;
  *(_QWORD *)(a1 + 24) = v24 + v23;
  *(_BYTE *)(a1 + 32) = 1;
  return a1;
}
