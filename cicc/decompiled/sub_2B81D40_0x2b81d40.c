// Function: sub_2B81D40
// Address: 0x2b81d40
//
__int64 __fastcall sub_2B81D40(__int64 a1, __int64 a2, int *a3)
{
  __int64 v6; // r9
  char v7; // cl
  __int64 v8; // rdx
  int v9; // esi
  int v10; // r8d
  int v11; // r15d
  int *v12; // r14
  unsigned int i; // eax
  int *v14; // r10
  int v15; // r11d
  unsigned int v16; // eax
  unsigned int v17; // esi
  unsigned int v18; // eax
  int v19; // edx
  unsigned int v20; // edi
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  char v24; // al
  int *v26; // [rsp+8h] [rbp-38h] BYREF

  v6 = *(_QWORD *)a2;
  v7 = *(_BYTE *)(a2 + 8) & 1;
  if ( v7 )
  {
    v8 = a2 + 16;
    v9 = 7;
  }
  else
  {
    v8 = *(_QWORD *)(a2 + 16);
    v17 = *(_DWORD *)(a2 + 24);
    if ( !v17 )
    {
      v18 = *(_DWORD *)(a2 + 8);
      v26 = 0;
      *(_QWORD *)a2 = v6 + 1;
      v19 = (v18 >> 1) + 1;
      goto LABEL_14;
    }
    v9 = v17 - 1;
  }
  v10 = a3[1];
  v11 = 1;
  v12 = 0;
  for ( i = v9
          & (((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v10) | ((unsigned __int64)(unsigned int)(37 * *a3) << 32))) >> 31)
           ^ (756364221 * v10)); ; i = v9 & v16 )
  {
    v14 = (int *)(v8 + 8LL * i);
    v15 = *v14;
    if ( *a3 == *v14 && v10 == v14[1] )
    {
      if ( v7 )
        v23 = v8 + 64;
      else
        v23 = 8LL * *(unsigned int *)(a2 + 24) + v8;
      v24 = 0;
      goto LABEL_22;
    }
    if ( v15 == -1 )
      break;
    if ( v15 == -2 && v14[1] == -2 && !v12 )
      v12 = (int *)(v8 + 8LL * i);
LABEL_10:
    v16 = v11 + i;
    ++v11;
  }
  if ( v14[1] != -1 )
    goto LABEL_10;
  v18 = *(_DWORD *)(a2 + 8);
  if ( !v12 )
    v12 = v14;
  *(_QWORD *)a2 = v6 + 1;
  v26 = v12;
  v19 = (v18 >> 1) + 1;
  if ( v7 )
  {
    v20 = 24;
    v17 = 8;
    goto LABEL_15;
  }
  v17 = *(_DWORD *)(a2 + 24);
LABEL_14:
  v20 = 3 * v17;
LABEL_15:
  if ( 4 * v19 >= v20 )
  {
    v17 *= 2;
  }
  else if ( v17 - *(_DWORD *)(a2 + 12) - v19 > v17 >> 3 )
  {
    goto LABEL_17;
  }
  sub_2B81830(a2, v17);
  sub_2B45390(a2, a3, &v26);
  v18 = *(_DWORD *)(a2 + 8);
LABEL_17:
  v14 = v26;
  *(_DWORD *)(a2 + 8) = (2 * (v18 >> 1) + 2) | v18 & 1;
  if ( *v14 != -1 || v14[1] != -1 )
    --*(_DWORD *)(a2 + 12);
  *v14 = *a3;
  v14[1] = a3[1];
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v21 = a2 + 16;
    v22 = 64;
  }
  else
  {
    v21 = *(_QWORD *)(a2 + 16);
    v22 = 8LL * *(unsigned int *)(a2 + 24);
  }
  v6 = *(_QWORD *)a2;
  v23 = v22 + v21;
  v24 = 1;
LABEL_22:
  *(_QWORD *)a1 = a2;
  *(_BYTE *)(a1 + 32) = v24;
  *(_QWORD *)(a1 + 8) = v6;
  *(_QWORD *)(a1 + 16) = v14;
  *(_QWORD *)(a1 + 24) = v23;
  return a1;
}
