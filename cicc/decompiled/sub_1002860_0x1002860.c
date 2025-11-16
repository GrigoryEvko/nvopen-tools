// Function: sub_1002860
// Address: 0x1002860
//
__int64 __fastcall sub_1002860(__int64 a1, unsigned int a2, unsigned int a3)
{
  unsigned int v3; // r13d
  __int64 v4; // rbx
  unsigned __int64 v5; // rax
  unsigned int v7; // ebx
  __int64 v8; // r13
  unsigned __int64 v9; // rax

  if ( a2 == 365 )
  {
    *(_DWORD *)(a1 + 8) = a3;
    if ( a3 > 0x40 )
    {
      sub_C43690(a1, -1, 1);
    }
    else
    {
      v9 = 0xFFFFFFFFFFFFFFFFLL >> -(char)a3;
      if ( !a3 )
        v9 = 0;
      *(_QWORD *)a1 = v9;
    }
    return a1;
  }
  if ( a2 > 0x16D )
  {
    if ( a2 != 366 )
      goto LABEL_27;
    *(_DWORD *)(a1 + 8) = a3;
    if ( a3 <= 0x40 )
    {
      *(_QWORD *)a1 = 0;
      return a1;
    }
    sub_C43690(a1, 0, 0);
    return a1;
  }
  if ( a2 != 329 )
  {
    if ( a2 == 330 )
    {
      v7 = a3 - 1;
      *(_DWORD *)(a1 + 8) = a3;
      v8 = 1LL << ((unsigned __int8)a3 - 1);
      if ( a3 > 0x40 )
      {
        sub_C43690(a1, 0, 0);
        if ( *(_DWORD *)(a1 + 8) > 0x40u )
        {
          *(_QWORD *)(*(_QWORD *)a1 + 8LL * (v7 >> 6)) |= v8;
          return a1;
        }
      }
      else
      {
        *(_QWORD *)a1 = 0;
      }
      *(_QWORD *)a1 |= v8;
      return a1;
    }
LABEL_27:
    BUG();
  }
  v3 = a3 - 1;
  *(_DWORD *)(a1 + 8) = a3;
  v4 = ~(1LL << ((unsigned __int8)a3 - 1));
  if ( a3 <= 0x40 )
  {
    v5 = 0xFFFFFFFFFFFFFFFFLL >> -(char)a3;
    if ( !a3 )
      v5 = 0;
    *(_QWORD *)a1 = v5;
    goto LABEL_8;
  }
  sub_C43690(a1, -1, 1);
  if ( *(_DWORD *)(a1 + 8) <= 0x40u )
  {
LABEL_8:
    *(_QWORD *)a1 &= v4;
    return a1;
  }
  *(_QWORD *)(*(_QWORD *)a1 + 8LL * (v3 >> 6)) &= v4;
  return a1;
}
