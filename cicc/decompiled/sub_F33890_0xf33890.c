// Function: sub_F33890
// Address: 0xf33890
//
__int64 __fastcall sub_F33890(__int64 *a1, unsigned int a2)
{
  __int64 v2; // r8
  __int64 v3; // rsi
  _QWORD *v4; // rax
  _QWORD *v5; // rdx
  unsigned int v7; // r9d

  v2 = *a1;
  v3 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1[1] - 8LL) + 32LL * *(unsigned int *)(*(_QWORD *)a1[1] + 72LL) + 8LL * a2);
  if ( *(_BYTE *)(*a1 + 28) )
  {
    v4 = *(_QWORD **)(v2 + 8);
    v5 = &v4[*(unsigned int *)(v2 + 20)];
    if ( v4 == v5 )
    {
      return 0;
    }
    else
    {
      while ( v3 != *v4 )
      {
        if ( v5 == ++v4 )
          return 0;
      }
      return *(unsigned __int8 *)(*a1 + 28);
    }
  }
  else
  {
    LOBYTE(v7) = sub_C8CA60(v2, v3) != 0;
    return v7;
  }
}
