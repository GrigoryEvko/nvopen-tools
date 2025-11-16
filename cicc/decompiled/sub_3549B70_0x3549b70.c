// Function: sub_3549B70
// Address: 0x3549b70
//
__int64 __fastcall sub_3549B70(__int64 a1, int *a2, _QWORD *a3)
{
  __int64 v4; // r9
  int v5; // edi
  int v6; // eax
  unsigned int v7; // edx
  _DWORD *v8; // rcx
  int v9; // esi
  __int64 result; // rax
  int v11; // r11d
  _DWORD *v12; // r10

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v4 = a1 + 16;
    v5 = 3;
  }
  else
  {
    result = *(unsigned int *)(a1 + 24);
    v4 = *(_QWORD *)(a1 + 16);
    v5 = result - 1;
    if ( !(_DWORD)result )
    {
      *a3 = 0;
      return result;
    }
  }
  v6 = *a2;
  v7 = v5 & (37 * *a2);
  v8 = (_DWORD *)(v4 + 4LL * v7);
  v9 = *v8;
  if ( v6 == *v8 )
  {
    *a3 = v8;
    return 1;
  }
  else
  {
    v11 = 1;
    v12 = 0;
    while ( v9 != -1 )
    {
      if ( !v12 && v9 == -2 )
        v12 = v8;
      v7 = v5 & (v11 + v7);
      v8 = (_DWORD *)(v4 + 4LL * v7);
      v9 = *v8;
      if ( v6 == *v8 )
      {
        *a3 = v8;
        return 1;
      }
      ++v11;
    }
    if ( !v12 )
      v12 = v8;
    *a3 = v12;
    return 0;
  }
}
