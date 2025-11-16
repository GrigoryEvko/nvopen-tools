// Function: sub_9C9670
// Address: 0x9c9670
//
__int64 __fastcall sub_9C9670(__int64 a1, __int64 *a2)
{
  char v3; // dl
  __int64 v4; // rsi
  int v5; // ecx
  __int64 v6; // r8
  __int64 v7; // r9
  int v8; // ebx
  unsigned __int64 v9; // rax
  unsigned int i; // eax
  __int64 v11; // r10
  unsigned int v12; // eax
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v16; // r10

  v3 = *(_BYTE *)(a1 + 8) & 1;
  if ( v3 )
  {
    v4 = a1 + 16;
    v5 = 3;
  }
  else
  {
    v13 = *(unsigned int *)(a1 + 24);
    v4 = *(_QWORD *)(a1 + 16);
    if ( !(_DWORD)v13 )
    {
LABEL_16:
      v16 = 24 * v13;
LABEL_17:
      v11 = v4 + v16;
      goto LABEL_10;
    }
    v5 = v13 - 1;
  }
  v6 = *a2;
  v7 = a2[1];
  v8 = 1;
  v9 = 0xBF58476D1CE4E5B9LL
     * (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)
      | ((unsigned __int64)(((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4)) << 32));
  for ( i = v5 & ((v9 >> 31) ^ v9); ; i = v5 & v12 )
  {
    v11 = v4 + 24LL * i;
    if ( *(_QWORD *)v11 == v6 && *(_QWORD *)(v11 + 8) == v7 )
      break;
    if ( *(_QWORD *)v11 == -4096 && *(_QWORD *)(v11 + 8) == -4096 )
    {
      if ( !v3 )
      {
        v13 = *(unsigned int *)(a1 + 24);
        goto LABEL_16;
      }
      v16 = 96;
      goto LABEL_17;
    }
    v12 = v8 + i;
    ++v8;
  }
LABEL_10:
  v14 = 96;
  if ( !v3 )
    v14 = 24LL * *(unsigned int *)(a1 + 24);
  if ( v11 == v4 + v14 )
    return 0;
  else
    return *(_QWORD *)(*(_QWORD *)(a1 + 112) + 24LL * *(unsigned int *)(v11 + 16) + 16);
}
