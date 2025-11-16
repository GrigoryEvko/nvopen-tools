// Function: sub_2448070
// Address: 0x2448070
//
__int64 __fastcall sub_2448070(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  int v3; // eax
  __int64 v4; // rax
  __int64 v5; // r8
  __int64 result; // rax
  int v7; // ecx
  unsigned __int64 v8; // rdx
  unsigned __int64 *v9; // rsi
  __int64 v10; // rdi
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // rdx
  __int64 v15; // rax
  int v16; // edx
  unsigned __int64 v17; // r8

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
    v9 = (unsigned __int64 *)(a2 + 16);
    v10 = a1 + 272;
    while ( 1 )
    {
      while ( 1 )
      {
        v12 = *v9;
        v13 = *(_QWORD *)result;
        if ( *v9 == -1 )
          break;
        if ( v12 == -2 )
        {
          *(_QWORD *)result = -2;
          *v9 = v13;
          if ( v13 <= 0xFFFFFFFFFFFFFFFDLL )
            goto LABEL_17;
        }
        else
        {
          *(_QWORD *)result = v12;
          if ( v13 <= 0xFFFFFFFFFFFFFFFDLL )
          {
            v17 = *(_QWORD *)(result + 8);
            *(_QWORD *)(result + 8) = v9[1];
            *v9 = v13;
            v9[1] = v17;
          }
          else
          {
            v11 = v9[1];
            *v9 = v13;
            *(_QWORD *)(result + 8) = v11;
          }
        }
LABEL_14:
        result += 16;
        v9 += 2;
        if ( v10 == result )
          return result;
      }
      *(_QWORD *)result = -1;
      *v9 = v13;
      if ( v13 > 0xFFFFFFFFFFFFFFFDLL )
        goto LABEL_14;
LABEL_17:
      v14 = *(_QWORD *)(result + 8);
      result += 16;
      v9 += 2;
      *(v9 - 1) = v14;
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
    v8 = *(_QWORD *)(a1 + result);
    *(_QWORD *)(a2 + result) = v8;
    if ( v8 <= 0xFFFFFFFFFFFFFFFDLL )
      *(_QWORD *)(a2 + result + 8) = *(_QWORD *)(a1 + result + 8);
    result += 16;
  }
  while ( result != 272 );
  *(_BYTE *)(a1 + 8) &= ~1u;
  *(_QWORD *)(a1 + 16) = v5;
  *(_DWORD *)(a1 + 24) = v7;
  return result;
}
