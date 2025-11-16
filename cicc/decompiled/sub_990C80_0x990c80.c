// Function: sub_990C80
// Address: 0x990c80
//
__int64 __fastcall sub_990C80(__int64 a1, unsigned int a2, unsigned int a3)
{
  unsigned int v3; // ebx
  __int64 v4; // r13
  unsigned __int64 v6; // rax
  unsigned int v7; // r13d
  __int64 v8; // rbx
  unsigned __int64 v9; // rax

  if ( a2 == 3 )
  {
    v7 = a3 - 1;
    *(_DWORD *)(a1 + 8) = a3;
    v8 = ~(1LL << ((unsigned __int8)a3 - 1));
    if ( a3 > 0x40 )
    {
      sub_C43690(a1, -1, 1);
      if ( *(_DWORD *)(a1 + 8) > 0x40u )
      {
        *(_QWORD *)(*(_QWORD *)a1 + 8LL * (v7 >> 6)) &= v8;
        return a1;
      }
    }
    else
    {
      v9 = 0xFFFFFFFFFFFFFFFFLL >> -(char)a3;
      if ( !a3 )
        v9 = 0;
      *(_QWORD *)a1 = v9;
    }
    *(_QWORD *)a1 &= v8;
    return a1;
  }
  if ( a2 > 3 )
  {
    if ( a2 != 4 )
      goto LABEL_27;
    *(_DWORD *)(a1 + 8) = a3;
    if ( a3 <= 0x40 )
    {
      v6 = 0xFFFFFFFFFFFFFFFFLL >> -(char)a3;
      if ( !a3 )
        v6 = 0;
      *(_QWORD *)a1 = v6;
      return a1;
    }
    sub_C43690(a1, -1, 1);
    return a1;
  }
  if ( a2 != 1 )
  {
    if ( a2 == 2 )
    {
      *(_DWORD *)(a1 + 8) = a3;
      if ( a3 <= 0x40 )
      {
        *(_QWORD *)a1 = 0;
        return a1;
      }
      sub_C43690(a1, 0, 0);
      return a1;
    }
LABEL_27:
    BUG();
  }
  v3 = a3 - 1;
  *(_DWORD *)(a1 + 8) = a3;
  v4 = 1LL << ((unsigned __int8)a3 - 1);
  if ( a3 <= 0x40 )
  {
    *(_QWORD *)a1 = 0;
LABEL_6:
    *(_QWORD *)a1 |= v4;
    return a1;
  }
  sub_C43690(a1, 0, 0);
  if ( *(_DWORD *)(a1 + 8) <= 0x40u )
    goto LABEL_6;
  *(_QWORD *)(*(_QWORD *)a1 + 8LL * (v3 >> 6)) |= v4;
  return a1;
}
