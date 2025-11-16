// Function: sub_36F0410
// Address: 0x36f0410
//
__int64 __fastcall sub_36F0410(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  int v3; // eax
  __int64 v4; // rax
  __int64 v5; // r8
  __int64 result; // rax
  int v7; // ecx
  unsigned __int8 v8; // dl
  unsigned __int8 *v9; // rsi
  __int64 v10; // rdi
  int v11; // edx
  unsigned __int8 v12; // dl
  unsigned __int8 v13; // cl
  int v14; // edx
  __int64 v15; // rax
  int v16; // edx
  int v17; // r8d

  v2 = *(_DWORD *)(a2 + 8) & 0xFFFFFFFE;
  *(_DWORD *)(a2 + 8) = *(_DWORD *)(a1 + 8) & 0xFFFFFFFE | *(_DWORD *)(a2 + 8) & 1;
  *(_DWORD *)(a1 + 8) = v2 | *(_DWORD *)(a1 + 8) & 1;
  v3 = *(_DWORD *)(a1 + 12);
  *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
  *(_DWORD *)(a2 + 12) = v3;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    if ( (*(_BYTE *)(a2 + 8) & 1) == 0 )
      goto LABEL_4;
    result = a1 + 16;
    v9 = (unsigned __int8 *)(a2 + 16);
    v10 = a1 + 80;
    while ( 1 )
    {
      while ( 1 )
      {
        v12 = *v9;
        v13 = *(_BYTE *)result;
        if ( *v9 == 0xFF )
          break;
        if ( v12 == 0xFE )
        {
          *(_BYTE *)result = -2;
          *v9 = v13;
          if ( v13 <= 0xFDu )
            goto LABEL_17;
        }
        else
        {
          *(_BYTE *)result = v12;
          if ( v13 <= 0xFDu )
          {
            v17 = *(_DWORD *)(result + 4);
            *(_DWORD *)(result + 4) = *((_DWORD *)v9 + 1);
            *v9 = v13;
            *((_DWORD *)v9 + 1) = v17;
          }
          else
          {
            v11 = *((_DWORD *)v9 + 1);
            *v9 = v13;
            *(_DWORD *)(result + 4) = v11;
          }
        }
LABEL_14:
        result += 8;
        v9 += 8;
        if ( v10 == result )
          return result;
      }
      *(_BYTE *)result = -1;
      *v9 = v13;
      if ( v13 > 0xFDu )
        goto LABEL_14;
LABEL_17:
      v14 = *(_DWORD *)(result + 4);
      result += 8;
      v9 += 8;
      *((_DWORD *)v9 - 1) = v14;
      if ( v10 == result )
        return result;
    }
  }
  if ( (*(_BYTE *)(a2 + 8) & 1) == 0 )
  {
    v15 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
    v16 = *(_DWORD *)(a2 + 24);
    *(_QWORD *)(a2 + 16) = v15;
    result = *(unsigned int *)(a1 + 24);
    *(_DWORD *)(a1 + 24) = v16;
    *(_DWORD *)(a2 + 24) = result;
    return result;
  }
  v4 = a2;
  a2 = a1;
  a1 = v4;
LABEL_4:
  *(_BYTE *)(a2 + 8) |= 1u;
  v5 = *(_QWORD *)(a2 + 16);
  result = 16;
  v7 = *(_DWORD *)(a2 + 24);
  do
  {
    v8 = *(_BYTE *)(a1 + result);
    *(_BYTE *)(a2 + result) = v8;
    if ( v8 <= 0xFDu )
      *(_DWORD *)(a2 + result + 4) = *(_DWORD *)(a1 + result + 4);
    result += 8;
  }
  while ( result != 80 );
  *(_BYTE *)(a1 + 8) &= ~1u;
  *(_QWORD *)(a1 + 16) = v5;
  *(_DWORD *)(a1 + 24) = v7;
  return result;
}
