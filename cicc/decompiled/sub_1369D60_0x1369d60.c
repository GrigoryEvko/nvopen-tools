// Function: sub_1369D60
// Address: 0x1369d60
//
__int64 __fastcall sub_1369D60(__int64 *a1, unsigned int a2)
{
  __int64 *v4; // rsi
  unsigned int v5; // r15d
  __int64 *v6; // rdx
  __int64 *v7; // r12
  unsigned int v8; // eax
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 result; // rax
  __int64 v12; // rax

  v4 = a1 + 1;
  v5 = a2 >> 7;
  v6 = (__int64 *)a1[1];
  if ( a1 + 1 == v6 )
  {
    v12 = sub_22077B0(40);
    *(_DWORD *)(v12 + 16) = v5;
    v9 = v12;
    *(_OWORD *)(v12 + 24) = 0;
    sub_2208C80(v12, v4);
    ++a1[3];
    goto LABEL_16;
  }
  v7 = (__int64 *)*a1;
  if ( v4 == (__int64 *)*a1 )
  {
    v7 = (__int64 *)a1[2];
    *a1 = (__int64)v7;
    v8 = *((_DWORD *)v7 + 4);
    if ( v5 == v8 )
      goto LABEL_11;
  }
  else
  {
    v8 = *((_DWORD *)v7 + 4);
    if ( v5 == v8 )
      goto LABEL_12;
  }
  if ( v5 < v8 )
  {
    if ( v6 != v7 )
    {
      do
        v7 = (__int64 *)v7[1];
      while ( v6 != v7 && v5 < *((_DWORD *)v7 + 4) );
    }
  }
  else if ( v4 != v7 )
  {
    while ( v5 > v8 )
    {
      v7 = (__int64 *)*v7;
      if ( v4 == v7 )
        break;
      v8 = *((_DWORD *)v7 + 4);
    }
  }
  *a1 = (__int64)v7;
LABEL_11:
  if ( v4 == v7 )
  {
LABEL_15:
    v10 = sub_22077B0(40);
    *(_DWORD *)(v10 + 16) = v5;
    v9 = v10;
    *(_OWORD *)(v10 + 24) = 0;
    sub_2208C80(v10, v7);
    ++a1[3];
    goto LABEL_16;
  }
LABEL_12:
  v9 = (__int64)v7;
  if ( v5 != *((_DWORD *)v7 + 4) )
  {
    if ( v5 > *((_DWORD *)v7 + 4) )
      v7 = (__int64 *)*v7;
    goto LABEL_15;
  }
LABEL_16:
  *a1 = v9;
  result = (a2 >> 6) & 1;
  *(_QWORD *)(v9 + 8 * result + 24) |= 1LL << a2;
  return result;
}
