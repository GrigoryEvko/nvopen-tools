// Function: sub_2537D10
// Address: 0x2537d10
//
__int64 __fastcall sub_2537D10(__int64 a1, __int64 a2)
{
  unsigned int v2; // r9d
  __int64 v3; // rcx
  __int64 v4; // rdi
  unsigned int v5; // r8d
  __int64 v6; // rdx
  _QWORD *v7; // rax
  __int64 v8; // rdx
  _QWORD *v9; // r11
  __int64 v10; // r10
  __int64 v11; // rdx
  _QWORD *v12; // rdx
  int v13; // eax
  __int64 v14; // r10
  int v15; // eax
  unsigned int v16; // r11d
  __int64 v17; // rdx
  int v19; // ebx

  v2 = *(unsigned __int8 *)(a1 + 97);
  if ( !(_BYTE)v2 )
    return 0;
  v3 = *(_QWORD *)(a1 + 136);
  v4 = v3 + 16LL * *(unsigned int *)(a1 + 144);
  if ( v3 == v4 )
    return 0;
  v5 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  while ( 1 )
  {
    v6 = *(_QWORD *)(v3 + 8);
    if ( *(_DWORD *)(v6 + 12) != 2 )
      break;
    v3 += 16;
    if ( v4 == v3 )
      return 0;
  }
  if ( *(_DWORD *)(v6 + 40) )
    goto LABEL_17;
LABEL_6:
  v7 = *(_QWORD **)(v6 + 56);
  v8 = 8LL * *(unsigned int *)(v6 + 64);
  v9 = &v7[(unsigned __int64)v8 / 8];
  v10 = v8 >> 3;
  v11 = v8 >> 5;
  if ( v11 )
  {
    v12 = &v7[4 * v11];
    while ( a2 != *v7 )
    {
      if ( a2 == v7[1] )
      {
        ++v7;
        goto LABEL_13;
      }
      if ( a2 == v7[2] )
      {
        v7 += 2;
        goto LABEL_13;
      }
      if ( a2 == v7[3] )
      {
        v7 += 3;
        goto LABEL_13;
      }
      v7 += 4;
      if ( v12 == v7 )
      {
        v10 = v9 - v7;
        goto LABEL_24;
      }
    }
    goto LABEL_13;
  }
LABEL_24:
  if ( v10 != 2 )
  {
    if ( v10 != 3 )
    {
      if ( v10 != 1 || a2 != *v7 )
        goto LABEL_14;
LABEL_13:
      if ( v9 == v7 )
        goto LABEL_14;
      return v2;
    }
    if ( a2 == *v7 )
      goto LABEL_13;
    ++v7;
  }
  if ( a2 == *v7 )
    goto LABEL_13;
  if ( a2 == *++v7 )
    goto LABEL_13;
LABEL_14:
  while ( 1 )
  {
    v3 += 16;
    if ( v4 == v3 )
      return 0;
    v6 = *(_QWORD *)(v3 + 8);
    if ( *(_DWORD *)(v6 + 12) != 2 )
    {
      if ( !*(_DWORD *)(v6 + 40) )
        goto LABEL_6;
LABEL_17:
      v13 = *(_DWORD *)(v6 + 48);
      v14 = *(_QWORD *)(v6 + 32);
      if ( v13 )
      {
        v15 = v13 - 1;
        v16 = v15 & v5;
        v17 = *(_QWORD *)(v14 + 8LL * (v15 & v5));
        if ( a2 == v17 )
          return v2;
        v19 = 1;
        while ( v17 != -4096 )
        {
          v16 = v15 & (v19 + v16);
          v17 = *(_QWORD *)(v14 + 8LL * v16);
          if ( a2 == v17 )
            return v2;
          ++v19;
        }
      }
    }
  }
}
