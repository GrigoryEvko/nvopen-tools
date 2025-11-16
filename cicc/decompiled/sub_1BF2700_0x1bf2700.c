// Function: sub_1BF2700
// Address: 0x1bf2700
//
__int64 __fastcall sub_1BF2700(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r8
  unsigned int v5; // r9d
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // rdi
  int v9; // edx
  int v10; // r10d

  if ( !a2 )
    return 0;
  if ( *(_BYTE *)(a2 + 16) != 77 )
    return 0;
  v3 = *(unsigned int *)(a1 + 128);
  v4 = *(_QWORD *)(a1 + 112);
  if ( !(_DWORD)v3 )
    return 0;
  v5 = v3 - 1;
  v6 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (__int64 *)(v4 + 16LL * v6);
  v8 = *v7;
  if ( a2 != *v7 )
  {
    v9 = 1;
    while ( v8 != -8 )
    {
      v10 = v9 + 1;
      v6 = v5 & (v9 + v6);
      v7 = (__int64 *)(v4 + 16LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
        goto LABEL_6;
      v9 = v10;
    }
    return 0;
  }
LABEL_6:
  LOBYTE(v5) = v7 != (__int64 *)(v4 + 16 * v3);
  return v5;
}
