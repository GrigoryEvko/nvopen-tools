// Function: sub_2B17E30
// Address: 0x2b17e30
//
__int64 __fastcall sub_2B17E30(__int64 *a1, __int64 a2)
{
  __int64 v2; // r10
  __int64 v3; // rdi
  __int64 v4; // r9
  int v5; // r11d
  unsigned int v6; // eax
  __int64 v7; // rcx
  int v8; // edx
  unsigned int v9; // ecx
  __int64 v10; // r8
  int v11; // eax
  int v13; // eax
  int v14; // edx
  int v15; // eax

  v2 = *a1;
  v3 = *(_QWORD *)(a2 - 96);
  if ( (*(_BYTE *)(v2 + 88) & 1) == 0 )
  {
    v13 = *(_DWORD *)(v2 + 104);
    if ( !v13 )
      return *(_QWORD *)(a2 - 96);
    v8 = v13 - 1;
    v4 = *(_QWORD *)(v2 + 96);
    v5 = v8;
    v6 = (v13 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = *(_QWORD *)(v4 + 72LL * (v8 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))));
    if ( a2 == v7 )
      goto LABEL_3;
LABEL_7:
    v14 = 1;
    while ( v7 != -4096 )
    {
      v6 = v5 & (v14 + v6);
      v7 = *(_QWORD *)(v4 + 72LL * v6);
      if ( a2 == v7 )
      {
        if ( (*(_BYTE *)(v2 + 88) & 1) != 0 )
        {
          v8 = 3;
        }
        else
        {
          v15 = *(_DWORD *)(v2 + 104);
          v10 = 0;
          v8 = v15 - 1;
          if ( !v15 )
            return v10;
        }
        goto LABEL_3;
      }
      ++v14;
    }
    return *(_QWORD *)(a2 - 96);
  }
  v4 = v2 + 96;
  v5 = 3;
  v6 = ((unsigned __int8)((unsigned int)a2 >> 4) ^ (unsigned __int8)((unsigned int)a2 >> 9)) & 3;
  v7 = *(_QWORD *)(v2
                 + 96
                 + 72LL * (((unsigned __int8)((unsigned int)a2 >> 4) ^ (unsigned __int8)((unsigned int)a2 >> 9)) & 3));
  v8 = 3;
  if ( a2 != v7 )
    goto LABEL_7;
LABEL_3:
  v9 = v8 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v10 = *(_QWORD *)(v4 + 72LL * v9);
  v11 = 1;
  if ( v3 != v10 )
  {
    while ( v10 != -4096 )
    {
      v9 = v8 & (v11 + v9);
      v10 = *(_QWORD *)(v4 + 72LL * v9);
      if ( v3 == v10 )
        return v10;
      ++v11;
    }
    return 0;
  }
  return v10;
}
