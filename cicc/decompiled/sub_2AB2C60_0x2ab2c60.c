// Function: sub_2AB2C60
// Address: 0x2ab2c60
//
__int64 __fastcall sub_2AB2C60(__int64 a1, __int64 a2, __int64 i)
{
  char v3; // r10
  __int64 v4; // rcx
  __int64 v5; // rax
  unsigned int v6; // edx
  int v7; // ebx
  __int64 v8; // r8
  unsigned int v9; // edx
  __int64 v11; // rax
  int v12; // [rsp+0h] [rbp-18h]

  v12 = i;
  if ( *(_BYTE *)a2 == 85 )
  {
    v11 = *(_QWORD *)(a2 - 32);
    if ( v11 )
    {
      if ( !*(_BYTE *)v11
        && *(_QWORD *)(v11 + 24) == *(_QWORD *)(a2 + 80)
        && (*(_BYTE *)(v11 + 33) & 0x20) != 0
        && *(_DWORD *)(v11 + 36) == 291 )
      {
        return 0;
      }
    }
  }
  v3 = BYTE4(i);
  if ( BYTE4(i) )
  {
    v4 = *(unsigned int *)(a1 + 184);
    v5 = *(_QWORD *)(a1 + 168);
    if ( (_DWORD)v4 )
    {
      v6 = 37 * i - 1;
LABEL_6:
      v7 = 1;
      for ( i = ((_DWORD)v4 - 1) & v6; ; i = ((_DWORD)v4 - 1) & v9 )
      {
        v8 = v5 + 72LL * (unsigned int)i;
        if ( *(_DWORD *)v8 == v12 && v3 == *(_BYTE *)(v8 + 4) )
        {
          v5 += 72LL * (unsigned int)i;
          return sub_B19060(v5 + 8, a2, i, v4);
        }
        if ( *(_DWORD *)v8 == -1 && *(_BYTE *)(v8 + 4) )
          break;
        v9 = v7 + i;
        ++v7;
      }
      i = 9 * v4;
      v5 += 72 * v4;
    }
    return sub_B19060(v5 + 8, a2, i, v4);
  }
  if ( (_DWORD)i != 1 )
  {
    v4 = *(unsigned int *)(a1 + 184);
    v5 = *(_QWORD *)(a1 + 168);
    if ( (_DWORD)v4 )
    {
      v6 = 37 * i;
      goto LABEL_6;
    }
    return sub_B19060(v5 + 8, a2, i, v4);
  }
  return 1;
}
