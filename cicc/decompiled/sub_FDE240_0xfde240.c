// Function: sub_FDE240
// Address: 0xfde240
//
__int64 __fastcall sub_FDE240(_QWORD *a1, unsigned int a2)
{
  unsigned int v2; // r15d
  _QWORD *v3; // rdx
  __int64 v4; // r12
  unsigned int v5; // eax
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 result; // rax
  __int64 v9; // rax

  v2 = a2 >> 7;
  v3 = (_QWORD *)*a1;
  if ( a1 == (_QWORD *)*a1 )
  {
    v9 = sub_22077B0(40);
    *(_DWORD *)(v9 + 16) = v2;
    v6 = v9;
    *(_OWORD *)(v9 + 24) = 0;
    sub_2208C80(v9, a1);
    ++a1[2];
    goto LABEL_16;
  }
  v4 = a1[3];
  if ( a1 == (_QWORD *)v4 )
  {
    v4 = a1[1];
    a1[3] = v4;
    v5 = *(_DWORD *)(v4 + 16);
    if ( v2 == v5 )
      goto LABEL_11;
  }
  else
  {
    v5 = *(_DWORD *)(v4 + 16);
    if ( v2 == v5 )
      goto LABEL_12;
  }
  if ( v2 < v5 )
  {
    if ( v3 != (_QWORD *)v4 )
    {
      do
        v4 = *(_QWORD *)(v4 + 8);
      while ( v3 != (_QWORD *)v4 && v2 < *(_DWORD *)(v4 + 16) );
    }
  }
  else if ( a1 != (_QWORD *)v4 )
  {
    while ( v2 > v5 )
    {
      v4 = *(_QWORD *)v4;
      if ( a1 == (_QWORD *)v4 )
        break;
      v5 = *(_DWORD *)(v4 + 16);
    }
  }
  a1[3] = v4;
LABEL_11:
  if ( a1 == (_QWORD *)v4 )
  {
LABEL_15:
    v7 = sub_22077B0(40);
    *(_DWORD *)(v7 + 16) = v2;
    v6 = v7;
    *(_OWORD *)(v7 + 24) = 0;
    sub_2208C80(v7, v4);
    ++a1[2];
    goto LABEL_16;
  }
LABEL_12:
  v6 = v4;
  if ( v2 != *(_DWORD *)(v4 + 16) )
  {
    if ( v2 > *(_DWORD *)(v4 + 16) )
      v4 = *(_QWORD *)v4;
    goto LABEL_15;
  }
LABEL_16:
  a1[3] = v6;
  result = (a2 >> 6) & 1;
  *(_QWORD *)(v6 + 8 * result + 24) |= 1LL << a2;
  return result;
}
