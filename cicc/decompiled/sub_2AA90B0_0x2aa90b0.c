// Function: sub_2AA90B0
// Address: 0x2aa90b0
//
bool __fastcall sub_2AA90B0(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // rdi
  __int64 v4; // rsi
  int v6; // edx
  __int64 v7; // r9
  int v8; // r10d
  int v9; // edx
  unsigned int i; // eax
  __int64 v11; // rdi
  unsigned int v12; // eax
  __int64 v13; // [rsp+8h] [rbp-8h] BYREF

  v2 = *a2;
  v13 = *a1;
  if ( *(_DWORD *)(v2 + 64) )
  {
    v6 = *(_DWORD *)(v2 + 72);
    v7 = *(_QWORD *)(v2 + 56);
    if ( v6 )
    {
      v8 = 1;
      v9 = v6 - 1;
      for ( i = v9 & ((BYTE4(v13) == 0) + 37 * v13 - 1); ; i = v9 & v12 )
      {
        v11 = v7 + 8LL * i;
        if ( (_DWORD)v13 == *(_DWORD *)v11 && BYTE4(v13) == *(_BYTE *)(v11 + 4) )
          break;
        if ( *(_DWORD *)v11 == -1 && *(_BYTE *)(v11 + 4) )
          return 0;
        v12 = v8 + i;
        ++v8;
      }
      return 1;
    }
    else
    {
      return 0;
    }
  }
  else
  {
    v3 = *(_QWORD *)(v2 + 80);
    v4 = v3 + 8LL * *(unsigned int *)(v2 + 88);
    return v4 != sub_2AA8390(v3, v4, (int *)&v13);
  }
}
