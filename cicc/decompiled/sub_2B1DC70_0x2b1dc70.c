// Function: sub_2B1DC70
// Address: 0x2b1dc70
//
__int64 __fastcall sub_2B1DC70(__int64 *a1, _BYTE *a2)
{
  __int64 result; // rax
  __int64 v3; // rcx
  __int64 v4; // r9
  int v5; // edi
  unsigned int v6; // edx
  _BYTE *v7; // r8
  int v8; // eax
  int v9; // edi
  int v10; // eax
  __int64 v11; // rdi
  int v12; // ecx
  unsigned int v13; // edx
  _BYTE *v14; // r8
  int v15; // eax

  result = 0;
  if ( *a2 == 61 )
  {
    v3 = *a1;
    if ( (*(_BYTE *)(*a1 + 88) & 1) != 0 )
    {
      v4 = v3 + 96;
      v5 = 3;
    }
    else
    {
      v9 = *(_DWORD *)(v3 + 104);
      v4 = *(_QWORD *)(v3 + 96);
      if ( !v9 )
        goto LABEL_12;
      v5 = v9 - 1;
    }
    v6 = v5 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = *(_BYTE **)(v4 + 72LL * v6);
    result = 0;
    if ( v7 == a2 )
      return result;
    v8 = 1;
    while ( v7 != (_BYTE *)-4096LL )
    {
      v6 = v5 & (v8 + v6);
      v7 = *(_BYTE **)(v4 + 72LL * v6);
      if ( a2 == v7 )
        return 0;
      ++v8;
    }
LABEL_12:
    v10 = *(_DWORD *)(v3 + 2000);
    v11 = *(_QWORD *)(v3 + 1984);
    if ( v10 )
    {
      v12 = v10 - 1;
      v13 = (v10 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v14 = *(_BYTE **)(v11 + 8LL * v13);
      result = 0;
      if ( a2 == v14 )
        return result;
      v15 = 1;
      while ( v14 != (_BYTE *)-4096LL )
      {
        v13 = v12 & (v15 + v13);
        v14 = *(_BYTE **)(v11 + 8LL * v13);
        if ( a2 == v14 )
          return 0;
        ++v15;
      }
    }
    return 1;
  }
  return result;
}
