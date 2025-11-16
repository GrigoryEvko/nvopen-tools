// Function: sub_14A9AE0
// Address: 0x14a9ae0
//
__int64 __fastcall sub_14A9AE0(__int64 a1, _BYTE *a2, __int64 *a3, unsigned int a4)
{
  unsigned int v5; // ecx
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rax
  unsigned int v10; // esi
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned int v14; // esi
  unsigned __int64 v15; // rdx
  __int64 v16; // rax

  v5 = *((_DWORD *)a3 + 2);
  *(_DWORD *)(a1 + 8) = v5;
  if ( v5 > 0x40 )
  {
    sub_16A4FD0(a1, a3);
    v5 = *(_DWORD *)(a1 + 8);
    if ( v5 > 0x40 )
    {
      sub_16A7DC0(a1, a4);
      goto LABEL_6;
    }
  }
  else
  {
    *(_QWORD *)a1 = *a3;
  }
  v7 = 0;
  if ( a4 != v5 )
    v7 = (*(_QWORD *)a1 << a4) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v5);
  *(_QWORD *)a1 = v7;
LABEL_6:
  if ( a4 )
  {
    if ( a4 > 0x40 )
    {
      sub_16A5260(a1, 0, a4);
    }
    else
    {
      v8 = *(_QWORD *)a1;
      v9 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)a4);
      if ( *(_DWORD *)(a1 + 8) > 0x40u )
        *(_QWORD *)v8 |= v9;
      else
        *(_QWORD *)a1 = v8 | v9;
    }
  }
  if ( *a2 )
  {
    v10 = *((_DWORD *)a3 + 2);
    v11 = 1LL << ((unsigned __int8)v10 - 1);
    v12 = *a3;
    if ( v10 > 0x40 )
    {
      if ( (*(_QWORD *)(v12 + 8LL * ((v10 - 1) >> 6)) & v11) == 0 )
        return a1;
    }
    else if ( (v12 & v11) == 0 )
    {
      return a1;
    }
    v14 = *(_DWORD *)(a1 + 8);
    v15 = *(_QWORD *)a1;
    v16 = 1LL << ((unsigned __int8)v14 - 1);
    if ( v14 > 0x40 )
      *(_QWORD *)(v15 + 8LL * ((v14 - 1) >> 6)) |= v16;
    else
      *(_QWORD *)a1 = v15 | v16;
  }
  return a1;
}
