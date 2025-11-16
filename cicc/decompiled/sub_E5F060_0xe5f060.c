// Function: sub_E5F060
// Address: 0xe5f060
//
__int64 __fastcall sub_E5F060(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // r13
  unsigned int v3; // r15d
  _QWORD *v4; // r14
  unsigned int v5; // eax

  v1 = *(_QWORD *)(a1 + 40);
  v2 = v1 + 8LL * *(unsigned int *)(a1 + 48);
  if ( v2 == v1 )
  {
    return 0;
  }
  else
  {
    v3 = 0;
    do
    {
      v4 = **(_QWORD ***)(*(_QWORD *)v1 + 8LL);
      while ( v4 )
      {
        LOBYTE(v5) = sub_E5EFF0((__int64 *)a1, (__int64)v4);
        v4 = (_QWORD *)*v4;
        if ( (_BYTE)v5 )
          v3 = v5;
      }
      v1 += 8;
    }
    while ( v2 != v1 );
  }
  return v3;
}
