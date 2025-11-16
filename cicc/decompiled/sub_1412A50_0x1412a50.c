// Function: sub_1412A50
// Address: 0x1412a50
//
__int64 __fastcall sub_1412A50(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  int v4; // ecx
  __int64 result; // rax
  int v6; // r8d
  __int64 *v7; // r12
  __int64 v8; // rdx
  unsigned int v9; // eax

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v3 = a1 + 16;
    v4 = 3;
  }
  else
  {
    result = *(unsigned int *)(a1 + 24);
    v3 = *(_QWORD *)(a1 + 16);
    if ( !(_DWORD)result )
      return result;
    v4 = result - 1;
  }
  result = v4 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v6 = 1;
  v7 = (__int64 *)(v3 + 88 * result);
  v8 = *v7;
  if ( *v7 == a2 )
  {
LABEL_4:
    if ( (v7[2] & 1) == 0 )
      j___libc_free_0(v7[3]);
    *v7 = -16;
    v9 = *(_DWORD *)(a1 + 8);
    ++*(_DWORD *)(a1 + 12);
    result = (2 * (v9 >> 1) - 2) | v9 & 1;
    *(_DWORD *)(a1 + 8) = result;
  }
  else
  {
    while ( v8 != -8 )
    {
      result = v4 & (unsigned int)(v6 + result);
      v7 = (__int64 *)(v3 + 88LL * (unsigned int)result);
      v8 = *v7;
      if ( *v7 == a2 )
        goto LABEL_4;
      ++v6;
    }
  }
  return result;
}
