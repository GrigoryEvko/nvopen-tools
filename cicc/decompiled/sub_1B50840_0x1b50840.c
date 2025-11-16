// Function: sub_1B50840
// Address: 0x1b50840
//
__int64 __fastcall sub_1B50840(__int64 a1, __int64 *a2, _QWORD *a3)
{
  __int64 v4; // r9
  int v5; // edi
  __int64 v6; // rcx
  unsigned int v7; // edx
  _QWORD *v8; // rax
  __int64 v9; // rsi
  __int64 result; // rax
  int v11; // r11d
  _QWORD *v12; // r10

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
  v7 = v5 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v8 = (_QWORD *)(v4 + 16LL * v7);
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
    while ( v9 != -8 )
    {
      if ( !v12 && v9 == -16 )
        v12 = v8;
      v7 = v5 & (v11 + v7);
      v8 = (_QWORD *)(v4 + 16LL * v7);
      v9 = *v8;
      if ( *v8 == v6 )
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
