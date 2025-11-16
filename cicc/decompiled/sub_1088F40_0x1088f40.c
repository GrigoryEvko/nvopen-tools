// Function: sub_1088F40
// Address: 0x1088f40
//
__int64 *__fastcall sub_1088F40(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 *v3; // r13
  int i; // eax
  __int64 v5; // rdx
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // rax
  __int64 *v10; // rbx
  __int64 *j; // r13
  __int64 v12; // rdx

  v2 = *(__int64 **)(a2 + 40);
  v3 = &v2[*(unsigned int *)(a2 + 48)];
  for ( i = *(_DWORD *)(a1 + 244); v3 != v2; i = *(_DWORD *)(a1 + 244) )
  {
    v5 = *v2;
    if ( i == 1 )
    {
      while ( 1 )
      {
        v7 = *(_QWORD *)(v5 + 136);
        if ( v7 <= 3 || *(_DWORD *)(*(_QWORD *)(v5 + 128) + v7 - 4) != 1870095406 )
          break;
        if ( v3 == ++v2 )
          goto LABEL_18;
        v5 = *v2;
      }
    }
    else if ( i == 2 )
    {
      while ( 1 )
      {
        v6 = *(_QWORD *)(v5 + 136);
        if ( v6 > 3 && *(_DWORD *)(*(_QWORD *)(v5 + 128) + v6 - 4) == 1870095406 )
          break;
        if ( v3 == ++v2 )
          goto LABEL_15;
        v5 = *v2;
      }
    }
    ++v2;
    sub_10860F0(a1, a2, v5);
  }
  if ( i != 2 )
  {
LABEL_18:
    v10 = *(__int64 **)(a2 + 56);
    for ( j = &v10[*(unsigned int *)(a2 + 64)]; j != v10; ++v10 )
    {
      v12 = *v10;
      if ( (*(_BYTE *)(*v10 + 8) & 2) == 0 || *(_BYTE *)(v12 + 12) == 3 )
        sub_1086E50(a1, (__int64 *)a2, v12);
    }
  }
LABEL_15:
  *(_DWORD *)(a1 + 40) = 0;
  v8 = *(_QWORD *)(a1 + 56) - *(_QWORD *)(a1 + 48);
  *(_BYTE *)(a1 + 240) = (unsigned __int64)v8 > 0x7F7F8;
  *(_DWORD *)(a1 + 28) = v8 >> 3;
  if ( (unsigned __int64)v8 > 0x3FFFFFFF8LL )
    sub_C64ED0("PE COFF object files can't have more than 2147483647 sections", 1u);
  return sub_1088350(a1);
}
