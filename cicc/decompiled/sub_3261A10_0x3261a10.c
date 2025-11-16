// Function: sub_3261A10
// Address: 0x3261a10
//
__int64 __fastcall sub_3261A10(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  unsigned __int16 *v5; // rdx
  unsigned __int16 v6; // ax
  __int64 v7; // r8
  __int64 v8; // rdx
  __int64 v10; // rcx
  int v11; // edx
  unsigned int *v12; // rcx

  v5 = *(unsigned __int16 **)(a1 + 48);
  v6 = *v5;
  v7 = *((_QWORD *)v5 + 1);
  v8 = 1;
  if ( v6 != 1 )
  {
    if ( !v6 )
      return 0;
    v8 = v6;
    if ( !*(_QWORD *)(a4 + 8LL * v6 + 112) )
      return 0;
  }
  if ( *(_BYTE *)(a4 + 500 * v8 + 6683) || (*(_BYTE *)(*(_QWORD *)a3 + 864LL) & 0x10) == 0 )
    return 0;
  v10 = **(_QWORD **)(a1 + 40);
  v11 = *(_DWORD *)(a1 + 24);
  if ( v11 == 220 )
  {
    if ( *(_DWORD *)(v10 + 24) != 226 )
      return 0;
  }
  else if ( v11 != 221 || *(_DWORD *)(v10 + 24) != 227 )
  {
    return 0;
  }
  v12 = *(unsigned int **)(v10 + 40);
  if ( v6 != *(_WORD *)(*(_QWORD *)(*(_QWORD *)v12 + 48LL) + 16LL * v12[2]) )
    return 0;
  return sub_33FAF80(a3, 269, a2, v6, v7, a3, *(_OWORD *)v12);
}
