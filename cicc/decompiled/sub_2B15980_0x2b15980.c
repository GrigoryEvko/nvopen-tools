// Function: sub_2B15980
// Address: 0x2b15980
//
bool __fastcall sub_2B15980(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r11
  _BYTE *v5; // rax
  __int64 v7; // r10
  int v8; // r8d
  unsigned int v9; // edx
  _BYTE *v10; // r9
  int v11; // ebx
  int v12; // r8d
  int v13; // r8d

  if ( a2 != a1 )
  {
    v4 = a3 + 96;
    while ( 1 )
    {
      v5 = *(_BYTE **)(a1 + 24);
      if ( *v5 != 91 )
        return a2 == a1;
      if ( (*(_BYTE *)(a3 + 88) & 1) != 0 )
        break;
      v13 = *(_DWORD *)(a3 + 104);
      v7 = *(_QWORD *)(a3 + 96);
      if ( v13 )
      {
        v8 = v13 - 1;
        goto LABEL_7;
      }
      a1 = *(_QWORD *)(a1 + 8);
      if ( a2 == a1 )
        return a2 == a1;
    }
    v7 = a3 + 96;
    v8 = 3;
LABEL_7:
    v9 = v8 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v10 = *(_BYTE **)(v7 + 72LL * v9);
    if ( v5 != v10 )
    {
      v11 = 1;
      while ( v10 != (_BYTE *)-4096LL )
      {
        v9 = v8 & (v11 + v9);
        v10 = *(_BYTE **)(v7 + 72LL * v9);
        if ( v5 == v10 )
          return a2 == a1;
        ++v11;
      }
      while ( 1 )
      {
        a1 = *(_QWORD *)(a1 + 8);
        if ( a2 == a1 )
          break;
        v5 = *(_BYTE **)(a1 + 24);
        if ( *v5 != 91 )
          break;
        if ( (*(_BYTE *)(a3 + 88) & 1) != 0 )
        {
          v7 = v4;
          v8 = 3;
          goto LABEL_7;
        }
        v12 = *(_DWORD *)(a3 + 104);
        v7 = *(_QWORD *)(a3 + 96);
        if ( v12 )
        {
          v8 = v12 - 1;
          goto LABEL_7;
        }
      }
    }
    return a2 == a1;
  }
  return a2 == a1;
}
