// Function: sub_2493220
// Address: 0x2493220
//
__int64 __fastcall sub_2493220(__int64 a1, unsigned __int8 *a2)
{
  __int64 v3; // rcx
  __int64 v4; // r8
  unsigned int v5; // edx
  unsigned __int8 **v6; // rax
  unsigned __int8 *v7; // rdi
  int v8; // eax
  int v9; // r10d

  if ( *a2 <= 0x15u )
    return sub_2492FB0((_QWORD **)a1, a2);
  v3 = *(unsigned int *)(a1 + 32);
  v4 = *(_QWORD *)(a1 + 16);
  if ( !(_DWORD)v3 )
    goto LABEL_6;
  v5 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v6 = (unsigned __int8 **)(v4 + 16LL * v5);
  v7 = *v6;
  if ( a2 != *v6 )
  {
    v8 = 1;
    while ( v7 != (unsigned __int8 *)-4096LL )
    {
      v9 = v8 + 1;
      v5 = (v3 - 1) & (v8 + v5);
      v6 = (unsigned __int8 **)(v4 + 16LL * v5);
      v7 = *v6;
      if ( a2 == *v6 )
        return (__int64)v6[1];
      v8 = v9;
    }
LABEL_6:
    v6 = (unsigned __int8 **)(v4 + 16 * v3);
  }
  return (__int64)v6[1];
}
