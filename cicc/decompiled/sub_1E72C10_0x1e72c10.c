// Function: sub_1E72C10
// Address: 0x1e72c10
//
__int64 __fastcall sub_1E72C10(__int64 a1, __int64 a2)
{
  _DWORD *v3; // rdi
  __int64 (*v4)(); // rax
  int v5; // eax
  int v6; // edx
  _DWORD *v7; // rdi
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rdx
  unsigned __int16 *v12; // r13
  unsigned __int16 *v13; // r12
  __int64 v14; // r13

  v3 = *(_DWORD **)(a1 + 152);
  if ( v3[2] )
  {
    v4 = *(__int64 (**)())(*(_QWORD *)v3 + 24LL);
    if ( v4 != sub_1D00B90 )
    {
      if ( ((unsigned int (__fastcall *)(_DWORD *, __int64, _QWORD))v4)(v3, a2, 0) )
        return 1;
    }
  }
  v5 = sub_1F4BA40(*(_QWORD *)(a1 + 8), *(_QWORD *)(a2 + 8), 0);
  v6 = *(_DWORD *)(a1 + 168);
  if ( v6 )
  {
    v7 = *(_DWORD **)(a1 + 8);
    if ( (unsigned int)(v6 + v5) > *v7 )
      return 1;
    if ( *(_DWORD *)(a1 + 24) != 1 )
      goto LABEL_6;
    if ( (unsigned __int8)sub_1F4B960(v7, *(_QWORD *)(a2 + 8), 0) )
      return 1;
    if ( *(_DWORD *)(a1 + 24) != 1 )
    {
      v7 = *(_DWORD **)(a1 + 8);
LABEL_6:
      if ( (unsigned __int8)sub_1F4B9D0(v7, *(_QWORD *)(a2 + 8), 0) )
        return 1;
    }
  }
  if ( !(unsigned __int8)sub_1F4B670(*(_QWORD *)(a1 + 8)) || *(char *)(a2 + 229) >= 0 )
    return 0;
  v9 = *(_QWORD *)(a2 + 24);
  if ( !v9 )
  {
    v14 = *(_QWORD *)a1 + 632LL;
    if ( (unsigned __int8)sub_1F4B670(v14) )
    {
      v9 = sub_1F4B8B0(v14, *(_QWORD *)(a2 + 8));
      *(_QWORD *)(a2 + 24) = v9;
    }
    else
    {
      v9 = *(_QWORD *)(a2 + 24);
    }
  }
  v10 = *(unsigned __int16 *)(v9 + 2);
  v11 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 176LL) + 136LL);
  v12 = (unsigned __int16 *)(v11 + 4 * (v10 + *(unsigned __int16 *)(v9 + 4)));
  v13 = (unsigned __int16 *)(v11 + 4 * v10);
  if ( v13 == v12 )
    return 0;
  while ( *(_DWORD *)(a1 + 164) >= (unsigned int)sub_1E72BE0(a1, *v13, v13[1]) )
  {
    v13 += 2;
    if ( v12 == v13 )
      return 0;
  }
  return 1;
}
