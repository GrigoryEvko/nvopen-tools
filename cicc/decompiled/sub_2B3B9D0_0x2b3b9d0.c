// Function: sub_2B3B9D0
// Address: 0x2b3b9d0
//
bool __fastcall sub_2B3B9D0(_QWORD *a1, unsigned __int8 *a2)
{
  __int64 v4; // rdx
  __int64 v6; // rdi
  int v7; // ecx
  unsigned int v8; // eax
  unsigned __int8 *v9; // rsi
  char v10; // r8
  int v11; // ecx
  __int64 v12; // rdx
  int v13; // r8d

  if ( (unsigned int)*a2 - 12 <= 1 )
    return 0;
  v4 = *a1;
  if ( (*(_BYTE *)(*a1 + 88LL) & 1) != 0 )
  {
    v6 = v4 + 96;
    v7 = 3;
  }
  else
  {
    v11 = *(_DWORD *)(v4 + 104);
    v6 = *(_QWORD *)(v4 + 96);
    if ( !v11 )
      goto LABEL_10;
    v7 = v11 - 1;
  }
  v8 = v7 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = *(unsigned __int8 **)(v6 + 72LL * v8);
  if ( a2 == v9 )
    return 1;
  v13 = 1;
  while ( v9 != (unsigned __int8 *)-4096LL )
  {
    v8 = v7 & (v13 + v8);
    v9 = *(unsigned __int8 **)(v6 + 72LL * v8);
    if ( a2 == v9 )
      return 1;
    ++v13;
  }
LABEL_10:
  v10 = sub_98ED70(a2, *(_QWORD *)(v4 + 3328), 0, 0, 0);
  if ( v10 )
    return 1;
  v12 = a1[1];
  if ( !*(_QWORD *)(v12 + 184) )
    return v10;
  return sub_2B0E280(*((_QWORD *)a2 + 2), 0, v12);
}
