// Function: sub_158ACE0
// Address: 0x158ace0
//
__int64 __fastcall sub_158ACE0(__int64 a1, __int64 a2)
{
  unsigned int v3; // eax
  __int64 v4; // rsi
  unsigned int v5; // ebx
  unsigned int v6; // eax
  unsigned int v7; // r12d
  __int64 v8; // rbx
  unsigned int v10; // eax

  if ( sub_158A0B0(a2) )
    goto LABEL_6;
  if ( (int)sub_16AEA10(a2, a2 + 16) <= 0 )
    goto LABEL_12;
  v3 = *(_DWORD *)(a2 + 24);
  v4 = *(_QWORD *)(a2 + 16);
  v5 = v3 - 1;
  if ( v3 > 0x40 )
  {
    if ( (*(_QWORD *)(v4 + 8LL * (v5 >> 6)) & (1LL << v5)) == 0 || (unsigned int)sub_16A58A0(a2 + 16) != v5 )
      goto LABEL_6;
LABEL_12:
    v10 = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 8) = v10;
    if ( v10 > 0x40 )
      sub_16A4FD0(a1, a2);
    else
      *(_QWORD *)a1 = *(_QWORD *)a2;
    return a1;
  }
  if ( v4 == 1LL << v5 )
    goto LABEL_12;
LABEL_6:
  v6 = *(_DWORD *)(a2 + 8);
  v7 = v6 - 1;
  *(_DWORD *)(a1 + 8) = v6;
  v8 = 1LL << ((unsigned __int8)v6 - 1);
  if ( v6 <= 0x40 )
  {
    *(_QWORD *)a1 = 0;
  }
  else
  {
    sub_16A4EF0(a1, 0, 0);
    if ( *(_DWORD *)(a1 + 8) > 0x40u )
    {
      *(_QWORD *)(*(_QWORD *)a1 + 8LL * (v7 >> 6)) |= v8;
      return a1;
    }
  }
  *(_QWORD *)a1 |= v8;
  return a1;
}
