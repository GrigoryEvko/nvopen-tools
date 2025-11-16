// Function: sub_3794520
// Address: 0x3794520
//
__int64 __fastcall sub_3794520(__int64 a1, unsigned __int64 *a2, _QWORD *a3)
{
  __int64 result; // rax
  int v5; // ecx
  __int64 v6; // r11
  __int64 v7; // r9
  int v8; // ebx
  unsigned __int64 v9; // rdi
  int v10; // esi
  unsigned int i; // eax
  __int64 v12; // rdx
  unsigned int v13; // eax
  int v14; // r10d

  result = *(unsigned int *)(a1 + 24);
  if ( !(_DWORD)result )
  {
    *a3 = 0;
    return result;
  }
  v5 = result - 1;
  v6 = 0;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 1;
  v9 = *a2;
  v10 = *((_DWORD *)a2 + 2);
  for ( i = (result - 1) & (v10 + ((v9 >> 4) ^ (v9 >> 9))); ; i = v5 & v13 )
  {
    v12 = v7 + 16LL * i;
    if ( v9 == *(_QWORD *)v12 && v10 == *(_DWORD *)(v12 + 8) )
    {
      *a3 = v12;
      return 1;
    }
    if ( !*(_QWORD *)v12 )
      break;
LABEL_5:
    v13 = v8 + i;
    ++v8;
  }
  v14 = *(_DWORD *)(v12 + 8);
  if ( v14 != -1 )
  {
    if ( !v6 && v14 == -2 )
      v6 = v7 + 16LL * i;
    goto LABEL_5;
  }
  if ( !v6 )
    v6 = v7 + 16LL * i;
  *a3 = v6;
  return 0;
}
