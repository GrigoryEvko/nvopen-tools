// Function: sub_2B0D770
// Address: 0x2b0d770
//
__int64 __fastcall sub_2B0D770(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned int a6)
{
  __int64 v6; // rcx
  __int64 v7; // rdx
  _QWORD *v8; // rax
  __int64 v9; // rsi
  _QWORD *v10; // r8
  __int64 v11; // rcx

  v6 = 8 * a2 - 8;
  v7 = *(_QWORD *)(*a1 + 8LL);
  v8 = a1 + 1;
  v9 = v6 >> 3;
  v10 = (_QWORD *)((char *)a1 + v6 + 8);
  v11 = v6 >> 5;
  if ( v11 <= 0 )
  {
LABEL_11:
    if ( v9 != 2 )
    {
      if ( v9 != 3 )
      {
        a6 = 1;
        if ( v9 != 1 )
          return a6;
        goto LABEL_14;
      }
      if ( v7 != *(_QWORD *)(*v8 + 8LL) )
        goto LABEL_15;
      ++v8;
    }
    if ( v7 != *(_QWORD *)(*v8 + 8LL) )
      goto LABEL_15;
    ++v8;
LABEL_14:
    a6 = 1;
    if ( v7 != *(_QWORD *)(*v8 + 8LL) )
    {
LABEL_15:
      LOBYTE(a6) = v10 == v8;
      return a6;
    }
    return a6;
  }
  while ( 1 )
  {
    if ( v7 != *(_QWORD *)(*v8 + 8LL) )
    {
      LOBYTE(a6) = v8 == v10;
      return a6;
    }
    if ( v7 != *(_QWORD *)(v8[1] + 8LL) )
    {
      LOBYTE(a6) = v10 == v8 + 1;
      return a6;
    }
    if ( v7 != *(_QWORD *)(v8[2] + 8LL) )
    {
      LOBYTE(a6) = v10 == v8 + 2;
      return a6;
    }
    if ( v7 != *(_QWORD *)(v8[3] + 8LL) )
      break;
    v8 += 4;
    if ( v8 == &a1[4 * v11 + 1] )
    {
      v9 = v10 - v8;
      goto LABEL_11;
    }
  }
  LOBYTE(a6) = v10 == v8 + 3;
  return a6;
}
