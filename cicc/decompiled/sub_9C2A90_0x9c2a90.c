// Function: sub_9C2A90
// Address: 0x9c2a90
//
__int64 __fastcall sub_9C2A90(__int64 a1, int a2, unsigned int a3)
{
  __int64 v4; // rdx
  __int64 v5; // r9
  unsigned int v6; // edi
  __int64 v7; // rax
  unsigned int v8; // r8d
  int v10; // eax
  int v11; // r11d

  v4 = *(unsigned int *)(a1 + 576);
  v5 = *(_QWORD *)(a1 + 560);
  if ( (_DWORD)v4 )
  {
    v6 = (v4 - 1) & (37 * a2);
    v7 = v5 + 32LL * v6;
    v8 = *(_DWORD *)v7;
    if ( *(_DWORD *)v7 != a2 )
    {
      v10 = 1;
      while ( v8 != -1 )
      {
        v11 = v10 + 1;
        v6 = (v4 - 1) & (v10 + v6);
        v7 = v5 + 32LL * v6;
        v8 = *(_DWORD *)v7;
        if ( *(_DWORD *)v7 == a2 )
          goto LABEL_3;
        v10 = v11;
      }
      return v8;
    }
LABEL_3:
    if ( v7 != v5 + 32 * v4 )
    {
      v8 = -1;
      if ( a3 < *(_DWORD *)(v7 + 16) )
        return *(unsigned int *)(*(_QWORD *)(v7 + 8) + 4LL * a3);
      return v8;
    }
  }
  return 0xFFFFFFFFLL;
}
