// Function: sub_1B27130
// Address: 0x1b27130
//
__int64 __fastcall sub_1B27130(__int64 *a1, __int64 a2)
{
  __int64 v2; // rcx
  char v3; // r8
  __int64 v4; // rdi
  int v5; // r10d
  unsigned int v6; // r9d
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx
  int v10; // edx
  unsigned int v11; // eax
  __int64 v12; // rax
  __int64 v13; // rax
  int v14; // eax
  int v15; // r11d

  v2 = *a1;
  v3 = *(_BYTE *)(*a1 + 8) & 1;
  if ( v3 )
  {
    v4 = v2 + 16;
    v5 = 15;
  }
  else
  {
    v12 = *(unsigned int *)(v2 + 24);
    v4 = *(_QWORD *)(v2 + 16);
    if ( !(_DWORD)v12 )
      goto LABEL_14;
    v5 = v12 - 1;
  }
  v6 = v5 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = v4 + 16LL * v6;
  v8 = *(_QWORD *)result;
  if ( a2 == *(_QWORD *)result )
    goto LABEL_4;
  v14 = 1;
  while ( v8 != -8 )
  {
    v15 = v14 + 1;
    v6 = v5 & (v14 + v6);
    result = v4 + 16LL * v6;
    v8 = *(_QWORD *)result;
    if ( a2 == *(_QWORD *)result )
      goto LABEL_4;
    v14 = v15;
  }
  if ( v3 )
  {
    v13 = 256;
    goto LABEL_15;
  }
  v12 = *(unsigned int *)(v2 + 24);
LABEL_14:
  v13 = 16 * v12;
LABEL_15:
  result = v4 + v13;
LABEL_4:
  v9 = 256;
  if ( !v3 )
    v9 = 16LL * *(unsigned int *)(v2 + 24);
  if ( result != v9 + v4 )
  {
    v10 = *(_DWORD *)(result + 8);
    if ( v10 > 0 )
    {
      *(_DWORD *)(result + 8) = v10 - 1;
    }
    else
    {
      *(_QWORD *)result = -16;
      v11 = *(_DWORD *)(v2 + 8);
      ++*(_DWORD *)(v2 + 12);
      result = (2 * (v11 >> 1) - 2) | v11 & 1;
      *(_DWORD *)(v2 + 8) = result;
    }
  }
  return result;
}
