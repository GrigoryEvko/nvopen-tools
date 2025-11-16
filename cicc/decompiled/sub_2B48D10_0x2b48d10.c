// Function: sub_2B48D10
// Address: 0x2b48d10
//
__int64 __fastcall sub_2B48D10(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  int v3; // eax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rcx
  int v7; // edx
  __int64 result; // rax
  __int64 *v9; // rsi
  __int64 v10; // rdi
  __int64 v11; // rdx
  bool v12; // cl
  __int64 v13; // r8
  __int64 v14; // rax
  int v15; // edx
  int v16; // ecx

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
    v9 = (__int64 *)(a2 + 16);
    v10 = a1 + 48;
    while ( 1 )
    {
      v11 = *(_QWORD *)result;
      v12 = *(_QWORD *)result != -8192 && *(_QWORD *)result != -4096;
      v13 = *v9;
      if ( *v9 == -4096 )
      {
        *(_QWORD *)result = -4096;
        *v9 = v11;
        if ( v12 )
          goto LABEL_22;
      }
      else
      {
        if ( v13 != -8192 )
        {
          *(_QWORD *)result = v13;
          if ( v12 )
          {
            v16 = *(_DWORD *)(result + 8);
            *(_DWORD *)(result + 8) = *((_DWORD *)v9 + 2);
            *v9 = v11;
            *((_DWORD *)v9 + 2) = v16;
          }
          else
          {
            *v9 = v11;
            *(_DWORD *)(result + 8) = *((_DWORD *)v9 + 2);
          }
          goto LABEL_17;
        }
        *(_QWORD *)result = -8192;
        *v9 = v11;
        if ( v12 )
LABEL_22:
          *((_DWORD *)v9 + 2) = *(_DWORD *)(result + 8);
      }
LABEL_17:
      result += 16;
      v9 += 2;
      if ( v10 == result )
        return result;
    }
  }
  if ( (*(_BYTE *)(a2 + 8) & 1) == 0 )
  {
    v14 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
    v15 = *(_DWORD *)(a2 + 24);
    *(_QWORD *)(a2 + 16) = v14;
    result = *(unsigned int *)(a1 + 24);
    *(_DWORD *)(a1 + 24) = v15;
    *(_DWORD *)(a2 + 24) = result;
    return result;
  }
  v4 = a2;
  a2 = a1;
  a1 = v4;
LABEL_4:
  *(_BYTE *)(a2 + 8) |= 1u;
  v5 = *(_QWORD *)(a1 + 16);
  v6 = *(_QWORD *)(a2 + 16);
  v7 = *(_DWORD *)(a2 + 24);
  *(_QWORD *)(a2 + 16) = v5;
  if ( v5 != -8192 && v5 != -4096 )
    *(_DWORD *)(a2 + 24) = *(_DWORD *)(a1 + 24);
  result = *(_QWORD *)(a1 + 32);
  *(_QWORD *)(a2 + 32) = result;
  if ( result != -8192 && result != -4096 )
  {
    result = *(unsigned int *)(a1 + 40);
    *(_DWORD *)(a2 + 40) = result;
  }
  *(_BYTE *)(a1 + 8) &= ~1u;
  *(_QWORD *)(a1 + 16) = v6;
  *(_DWORD *)(a1 + 24) = v7;
  return result;
}
