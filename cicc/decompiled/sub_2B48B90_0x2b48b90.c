// Function: sub_2B48B90
// Address: 0x2b48b90
//
unsigned __int64 __fastcall sub_2B48B90(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  int v3; // eax
  __int64 v4; // rax
  unsigned __int64 v5; // rax
  __int64 v6; // rcx
  int v7; // edx
  unsigned __int64 result; // rax
  unsigned __int64 *v9; // rsi
  __int64 v10; // rdi
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rcx
  int v13; // edx
  __int64 v14; // rax
  int v15; // edx
  int v16; // r8d

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
    v10 = a1 + 48;
    while ( 1 )
    {
      v11 = *v9;
      v12 = *(_QWORD *)result;
      if ( *v9 == -1 )
      {
        *(_QWORD *)result = -1;
        *v9 = v12;
        if ( v12 <= 0xFFFFFFFFFFFFFFFDLL )
          goto LABEL_21;
      }
      else
      {
        if ( v11 != -2 )
        {
          *(_QWORD *)result = v11;
          if ( v12 <= 0xFFFFFFFFFFFFFFFDLL )
          {
            v16 = *(_DWORD *)(result + 8);
            *(_DWORD *)(result + 8) = *((_DWORD *)v9 + 2);
            *v9 = v12;
            *((_DWORD *)v9 + 2) = v16;
          }
          else
          {
            v13 = *((_DWORD *)v9 + 2);
            *v9 = v12;
            *(_DWORD *)(result + 8) = v13;
          }
          goto LABEL_13;
        }
        *(_QWORD *)result = -2;
        *v9 = v12;
        if ( v12 <= 0xFFFFFFFFFFFFFFFDLL )
LABEL_21:
          *((_DWORD *)v9 + 2) = *(_DWORD *)(result + 8);
      }
LABEL_13:
      result += 16LL;
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
  if ( v5 <= 0xFFFFFFFFFFFFFFFDLL )
  {
    *(_DWORD *)(a2 + 24) = *(_DWORD *)(a1 + 24);
    result = *(_QWORD *)(a1 + 32);
    *(_QWORD *)(a2 + 32) = result;
    if ( result > 0xFFFFFFFFFFFFFFFDLL )
      goto LABEL_6;
  }
  else
  {
    result = *(_QWORD *)(a1 + 32);
    *(_QWORD *)(a2 + 32) = result;
    if ( result > 0xFFFFFFFFFFFFFFFDLL )
    {
LABEL_6:
      *(_BYTE *)(a1 + 8) &= ~1u;
      *(_QWORD *)(a1 + 16) = v6;
      *(_DWORD *)(a1 + 24) = v7;
      return result;
    }
  }
  result = *(unsigned int *)(a1 + 40);
  *(_DWORD *)(a2 + 40) = result;
  *(_QWORD *)(a1 + 16) = v6;
  *(_BYTE *)(a1 + 8) &= ~1u;
  *(_DWORD *)(a1 + 24) = v7;
  return result;
}
