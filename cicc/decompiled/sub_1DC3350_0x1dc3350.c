// Function: sub_1DC3350
// Address: 0x1dc3350
//
__int64 __fastcall sub_1DC3350(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4)
{
  unsigned __int64 i; // rdx
  __int64 v9; // rcx
  __int64 v10; // rsi
  unsigned int v11; // r10d
  __int64 *v12; // rax
  __int64 v13; // rbx
  int v15; // eax
  int v16; // r12d

  for ( i = *(_QWORD *)(a4 + 16); (*(_BYTE *)(i + 46) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
    ;
  v9 = *(unsigned int *)(a1 + 384);
  v10 = *(_QWORD *)(a1 + 368);
  if ( !(_DWORD)v9 )
  {
LABEL_8:
    v12 = (__int64 *)(v10 + 16 * v9);
    return sub_1DB79D0(a3, ((*(_BYTE *)(a4 + 4) & 4) == 0 ? 4LL : 2LL) | v12[1] & 0xFFFFFFFFFFFFFFF8LL, a2);
  }
  v11 = (v9 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
  v12 = (__int64 *)(v10 + 16LL * v11);
  v13 = *v12;
  if ( i != *v12 )
  {
    v15 = 1;
    while ( v13 != -8 )
    {
      v16 = v15 + 1;
      v11 = (v9 - 1) & (v15 + v11);
      v12 = (__int64 *)(v10 + 16LL * v11);
      v13 = *v12;
      if ( i == *v12 )
        return sub_1DB79D0(a3, ((*(_BYTE *)(a4 + 4) & 4) == 0 ? 4LL : 2LL) | v12[1] & 0xFFFFFFFFFFFFFFF8LL, a2);
      v15 = v16;
    }
    goto LABEL_8;
  }
  return sub_1DB79D0(a3, ((*(_BYTE *)(a4 + 4) & 4) == 0 ? 4LL : 2LL) | v12[1] & 0xFFFFFFFFFFFFFFF8LL, a2);
}
