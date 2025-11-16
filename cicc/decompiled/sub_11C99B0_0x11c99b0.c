// Function: sub_11C99B0
// Address: 0x11c99b0
//
bool __fastcall sub_11C99B0(__int64 *a1, __int64 *a2, unsigned int a3)
{
  __int64 v4; // rsi
  __int64 v7; // rax
  __int64 v9; // rax
  __int64 v10; // rdi
  unsigned int v11; // ecx
  int *v12; // rdx
  int v13; // esi
  int v14; // edx
  int v15; // r9d

  if ( (a2[((unsigned __int64)a3 >> 6) + 1] & (1LL << a3)) != 0 )
    return 0;
  v4 = *a2;
  if ( (((int)*(unsigned __int8 *)(*a2 + (a3 >> 2)) >> (2 * (a3 & 3))) & 3) == 0 )
    return 0;
  if ( (((int)*(unsigned __int8 *)(*a2 + (a3 >> 2)) >> (2 * (a3 & 3))) & 3) == 3 )
  {
    v7 = sub_BA8B30((__int64)a1, (__int64)(&off_4977320)[2 * a3], qword_4977328[2 * a3]);
    if ( v7 )
      goto LABEL_5;
    return 1;
  }
  v9 = *(unsigned int *)(v4 + 160);
  v10 = *(_QWORD *)(v4 + 144);
  if ( (_DWORD)v9 )
  {
    v11 = (v9 - 1) & (37 * a3);
    v12 = (int *)(v10 + 40LL * v11);
    v13 = *v12;
    if ( a3 == *v12 )
      goto LABEL_10;
    v14 = 1;
    while ( v13 != -1 )
    {
      v15 = v14 + 1;
      v11 = (v9 - 1) & (v14 + v11);
      v12 = (int *)(v10 + 40LL * v11);
      v13 = *v12;
      if ( a3 == *v12 )
        goto LABEL_10;
      v14 = v15;
    }
  }
  v12 = (int *)(v10 + 40 * v9);
LABEL_10:
  v7 = sub_BA8B30((__int64)a1, *((_QWORD *)v12 + 1), *((_QWORD *)v12 + 2));
  if ( v7 )
  {
LABEL_5:
    if ( !*(_BYTE *)v7 )
      return sub_97FAA0(*a2, *(_QWORD *)(v7 + 24), a3, a1);
    return 0;
  }
  return 1;
}
