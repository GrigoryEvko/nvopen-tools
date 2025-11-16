// Function: sub_1A3F5B0
// Address: 0x1a3f5b0
//
__int64 __fastcall sub_1A3F5B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // r8
  unsigned int v4; // r9d
  unsigned int *v5; // rdi
  int v6; // eax
  int v7; // edx
  unsigned int v8; // ecx
  unsigned int *v9; // rsi
  int v11; // esi
  int v12; // r10d

  v2 = *(unsigned int *)(a1 + 504);
  v3 = *(_QWORD *)(a1 + 488);
  v4 = 0;
  v5 = (unsigned int *)(v3 + 4 * v2);
  if ( (_DWORD)v2 )
  {
    v6 = v2 - 1;
    v7 = *(unsigned __int8 *)(a2 + 16) - 24;
    v8 = v6 & (37 * v7);
    v9 = (unsigned int *)(v3 + 4LL * v8);
    v4 = *v9;
    if ( v7 == *v9 )
    {
LABEL_3:
      LOBYTE(v4) = v5 != v9;
    }
    else
    {
      v11 = 1;
      while ( v4 != -1 )
      {
        v12 = v11 + 1;
        v8 = v6 & (v11 + v8);
        v9 = (unsigned int *)(v3 + 4LL * v8);
        v4 = *v9;
        if ( v7 == *v9 )
          goto LABEL_3;
        v11 = v12;
      }
      return 0;
    }
  }
  return v4;
}
