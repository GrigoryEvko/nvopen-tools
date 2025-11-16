// Function: sub_7E4A80
// Address: 0x7e4a80
//
__int64 __fastcall sub_7E4A80(__int64 a1, __int64 **a2, _QWORD *a3)
{
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // rbx
  __int64 **v7; // rax
  __int64 result; // rax
  __int64 **v9; // rax
  __int64 **v10; // rax

  v4 = *(_QWORD *)(a1 + 72);
  if ( *(_BYTE *)(v4 + 24) == 1 && *(_BYTE *)(a1 + 56) == *(_BYTE *)(v4 + 56) )
  {
    sub_7E4A80(*(_QWORD *)(a1 + 72));
  }
  else
  {
    *a2 = (__int64 *)v4;
    *a3 = 0;
  }
  v5 = sub_7E1F20(*(_QWORD *)v4);
  v6 = sub_7E1F20(*(_QWORD *)a1);
  sub_7E3EE0(v5);
  sub_7E3EE0(v6);
  if ( *(_BYTE *)(a1 + 56) == 16 )
  {
    v9 = *(__int64 ***)(v5 + 168);
    do
    {
      do
        v9 = (__int64 **)*v9;
      while ( ((_BYTE)v9[12] & 3) == 0 );
    }
    while ( (__int64 *)v6 != v9[5] );
    if ( ((_BYTE)v9[12] & 2) != 0 )
    {
      v10 = *(__int64 ***)(sub_7E1F20(**a2) + 168);
      do
      {
        do
          v10 = (__int64 **)*v10;
        while ( (__int64 *)v6 != v10[5] );
      }
      while ( ((_BYTE)v10[12] & 2) == 0 );
      result = -(__int64)v10[13];
      *a3 = result;
    }
    else
    {
      result = (__int64)v9[13];
      *a3 -= result;
    }
  }
  else
  {
    v7 = *(__int64 ***)(v6 + 168);
    do
    {
      do
        v7 = (__int64 **)*v7;
      while ( ((_BYTE)v7[12] & 3) == 0 );
    }
    while ( (__int64 *)v5 != v7[5] );
    result = (__int64)v7[13];
    *a3 += result;
  }
  return result;
}
