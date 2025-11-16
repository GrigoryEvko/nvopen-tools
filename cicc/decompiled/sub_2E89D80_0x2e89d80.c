// Function: sub_2E89D80
// Address: 0x2e89d80
//
__int16 __fastcall sub_2E89D80(__int64 a1, unsigned int a2, __int64 a3)
{
  int v4; // r13d
  __int64 v5; // r8
  __int64 v7; // rbx
  __int64 v8; // r9
  char v9; // r15
  char v10; // r10
  int v11; // eax
  __int64 v12; // r12
  __int64 v13; // rcx
  char v14; // dl
  char v15; // r10
  __int16 result; // ax
  unsigned int v17; // [rsp+Ch] [rbp-44h]
  unsigned __int8 v18; // [rsp+12h] [rbp-3Eh]
  char v19; // [rsp+13h] [rbp-3Dh]

  v4 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
  if ( !v4 )
  {
    v14 = 0;
    v15 = 0;
    goto LABEL_19;
  }
  v5 = a2;
  v7 = 0;
  v8 = 0;
  v9 = 0;
  v10 = 0;
  do
  {
    while ( 1 )
    {
      v11 = v7;
      v12 = *(_QWORD *)(a1 + 32) + 40 * v7;
      if ( *(_BYTE *)v12 || (_DWORD)v5 != *(_DWORD *)(v12 + 8) )
        goto LABEL_4;
      if ( a3 )
      {
        v13 = *(unsigned int *)(a3 + 8);
        if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
        {
          v17 = v5;
          v18 = v8;
          v19 = v10;
          sub_C8D5F0(a3, (const void *)(a3 + 16), v13 + 1, 4u, v5, v8);
          v5 = v17;
          v8 = v18;
          v10 = v19;
          v13 = *(unsigned int *)(a3 + 8);
          v11 = v7;
        }
        *(_DWORD *)(*(_QWORD *)a3 + 4 * v13) = v11;
        ++*(_DWORD *)(a3 + 8);
      }
      if ( (*(_BYTE *)(v12 + 3) & 0x10) != 0 )
        break;
      v8 = !(*(_BYTE *)(v12 + 4) & 1) | (unsigned int)v8;
LABEL_4:
      if ( v4 == (_DWORD)++v7 )
        goto LABEL_17;
    }
    if ( (*(_DWORD *)v12 & 0xFFF00) == 0 )
    {
      v9 = 1;
      goto LABEL_4;
    }
    if ( (*(_BYTE *)(v12 + 4) & 1) != 0 )
      v9 = *(_BYTE *)(v12 + 4) & 1;
    else
      v10 = 1;
    ++v7;
  }
  while ( v4 != (_DWORD)v7 );
LABEL_17:
  v14 = v9 | v10;
  v15 = (v9 ^ 1) & v10;
  if ( (_BYTE)v8 )
    v15 = 1;
LABEL_19:
  LOBYTE(result) = v15;
  HIBYTE(result) = v14;
  return result;
}
