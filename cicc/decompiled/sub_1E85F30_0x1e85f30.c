// Function: sub_1E85F30
// Address: 0x1e85f30
//
__int64 __fastcall sub_1E85F30(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // r8
  unsigned int v4; // ecx
  __int64 *v5; // rax
  __int64 v6; // rdi
  int v8; // eax
  int v9; // r10d

  for ( ; (*(_BYTE *)(a2 + 46) & 4) != 0; a2 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL )
    ;
  v2 = *(unsigned int *)(a1 + 384);
  v3 = *(_QWORD *)(a1 + 368);
  if ( (_DWORD)v2 )
  {
    v4 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v5 = (__int64 *)(v3 + 16LL * v4);
    v6 = *v5;
    if ( a2 == *v5 )
      return v5[1];
    v8 = 1;
    while ( v6 != -8 )
    {
      v9 = v8 + 1;
      v4 = (v2 - 1) & (v8 + v4);
      v5 = (__int64 *)(v3 + 16LL * v4);
      v6 = *v5;
      if ( a2 == *v5 )
        return v5[1];
      v8 = v9;
    }
  }
  return *(_QWORD *)(v3 + 16 * v2 + 8);
}
