// Function: sub_161E590
// Address: 0x161e590
//
__int64 __fastcall sub_161E590(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  int v3; // ecx
  __int64 result; // rax
  __int64 *v5; // rdx
  __int64 v6; // r9
  unsigned int v7; // eax
  int v8; // edx
  int v9; // r10d

  if ( (*(_BYTE *)(a1 + 24) & 1) != 0 )
  {
    v2 = a1 + 32;
    v3 = 3;
  }
  else
  {
    result = *(unsigned int *)(a1 + 40);
    v2 = *(_QWORD *)(a1 + 32);
    if ( !(_DWORD)result )
      return result;
    v3 = result - 1;
  }
  result = v3 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v5 = (__int64 *)(v2 + 24 * result);
  v6 = *v5;
  if ( *v5 == a2 )
  {
LABEL_4:
    *v5 = -8;
    v7 = *(_DWORD *)(a1 + 24);
    ++*(_DWORD *)(a1 + 28);
    result = (2 * (v7 >> 1) - 2) | v7 & 1;
    *(_DWORD *)(a1 + 24) = result;
  }
  else
  {
    v8 = 1;
    while ( v6 != -4 )
    {
      v9 = v8 + 1;
      result = v3 & (unsigned int)(v8 + result);
      v5 = (__int64 *)(v2 + 24LL * (unsigned int)result);
      v6 = *v5;
      if ( *v5 == a2 )
        goto LABEL_4;
      v8 = v9;
    }
  }
  return result;
}
