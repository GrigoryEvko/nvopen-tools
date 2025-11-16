// Function: sub_14A99F0
// Address: 0x14a99f0
//
__int64 *__fastcall sub_14A99F0(__int64 *a1, _BYTE *a2, __int64 *a3, unsigned int a4)
{
  unsigned int v5; // ecx
  unsigned __int64 v7; // rax
  unsigned int v8; // esi
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned int v12; // esi
  __int64 v13; // rdx
  __int64 v14; // rax

  v5 = *((_DWORD *)a3 + 2);
  *((_DWORD *)a1 + 2) = v5;
  if ( v5 > 0x40 )
  {
    sub_16A4FD0(a1, a3);
    v5 = *((_DWORD *)a1 + 2);
    if ( v5 > 0x40 )
    {
      sub_16A7DC0(a1, a4);
      goto LABEL_6;
    }
  }
  else
  {
    *a1 = *a3;
  }
  v7 = 0;
  if ( a4 != v5 )
    v7 = (*a1 << a4) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v5);
  *a1 = v7;
LABEL_6:
  if ( *a2 )
  {
    v8 = *((_DWORD *)a3 + 2);
    v9 = 1LL << ((unsigned __int8)v8 - 1);
    v10 = *a3;
    if ( v8 > 0x40 )
    {
      if ( (*(_QWORD *)(v10 + 8LL * ((v8 - 1) >> 6)) & v9) == 0 )
        return a1;
    }
    else if ( (v10 & v9) == 0 )
    {
      return a1;
    }
    v12 = *((_DWORD *)a1 + 2);
    v13 = *a1;
    v14 = 1LL << ((unsigned __int8)v12 - 1);
    if ( v12 > 0x40 )
      *(_QWORD *)(v13 + 8LL * ((v12 - 1) >> 6)) |= v14;
    else
      *a1 = v13 | v14;
  }
  return a1;
}
