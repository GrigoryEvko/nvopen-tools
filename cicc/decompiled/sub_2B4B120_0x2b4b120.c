// Function: sub_2B4B120
// Address: 0x2b4b120
//
int *__fastcall sub_2B4B120(__int64 a1, int *a2)
{
  char v4; // dl
  __int64 v5; // r8
  int v6; // esi
  int v7; // edi
  int v8; // r13d
  int *v9; // r11
  unsigned int i; // eax
  int *v11; // r9
  int v12; // r10d
  unsigned int v13; // eax
  unsigned int v14; // esi
  unsigned int v15; // eax
  int v16; // ecx
  unsigned int v17; // edi
  int *v18; // rax
  int *result; // rax
  int *v20; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( v4 )
  {
    v5 = a1 + 16;
    v6 = 7;
  }
  else
  {
    v14 = *(_DWORD *)(a1 + 24);
    v5 = *(_QWORD *)(a1 + 16);
    if ( !v14 )
    {
      v15 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v20 = 0;
      v16 = (v15 >> 1) + 1;
      goto LABEL_14;
    }
    v6 = v14 - 1;
  }
  v7 = a2[1];
  v8 = 1;
  v9 = 0;
  for ( i = v6
          & (((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v7) | ((unsigned __int64)(unsigned int)(37 * *a2) << 32))) >> 31)
           ^ (756364221 * v7)); ; i = v6 & v13 )
  {
    v11 = (int *)(v5 + 12LL * i);
    v12 = *v11;
    if ( *a2 == *v11 && v7 == v11[1] )
      return v11 + 2;
    if ( v12 == -1 )
      break;
    if ( v12 == -2 && v11[1] == -2 && !v9 )
      v9 = (int *)(v5 + 12LL * i);
LABEL_10:
    v13 = v8 + i;
    ++v8;
  }
  if ( v11[1] != -1 )
    goto LABEL_10;
  v15 = *(_DWORD *)(a1 + 8);
  v17 = 24;
  v14 = 8;
  if ( !v9 )
    v9 = v11;
  ++*(_QWORD *)a1;
  v20 = v9;
  v16 = (v15 >> 1) + 1;
  if ( !v4 )
  {
    v14 = *(_DWORD *)(a1 + 24);
LABEL_14:
    v17 = 3 * v14;
  }
  if ( 4 * v16 >= v17 )
  {
    v14 *= 2;
  }
  else if ( v14 - *(_DWORD *)(a1 + 12) - v16 > v14 >> 3 )
  {
    goto LABEL_17;
  }
  sub_2B4ABD0(a1, v14);
  sub_2B3D400(a1, a2, &v20);
  v15 = *(_DWORD *)(a1 + 8);
LABEL_17:
  *(_DWORD *)(a1 + 8) = (2 * (v15 >> 1) + 2) | v15 & 1;
  v18 = v20;
  if ( *v20 != -1 || v20[1] != -1 )
    --*(_DWORD *)(a1 + 12);
  result = v18 + 2;
  *(result - 2) = *a2;
  *(_QWORD *)(result - 1) = (unsigned int)a2[1];
  return result;
}
