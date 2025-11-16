// Function: sub_325DDD0
// Address: 0x325ddd0
//
__int64 __fastcall sub_325DDD0(int **a1, unsigned int a2)
{
  unsigned int v2; // r9d
  int v3; // edx
  unsigned int v4; // ecx
  __int64 v5; // r8
  unsigned int v6; // esi
  _QWORD *v7; // rax
  __int64 v8; // rdi
  unsigned int v10; // ecx
  unsigned int v11; // esi
  __int64 v12; // r10
  unsigned int v13; // r8d
  __int64 v14; // rdi

  v2 = a2;
  v3 = **a1;
  if ( (_BYTE)a2 )
  {
    if ( v3 )
    {
      v4 = 0;
      v5 = *(_QWORD *)a1[3];
      v6 = (unsigned int)*a1[2] >> 3;
      v7 = *(_QWORD **)a1[1];
      v8 = (__int64)&v7[(unsigned int)(v3 - 1) + 1];
      while ( *v7 == v5 + v4 )
      {
        ++v7;
        v4 += v6;
        if ( (_QWORD *)v8 == v7 )
          return v2;
      }
      return 0;
    }
    return v2;
  }
  v10 = v3 - 1;
  if ( v3 )
  {
    v11 = 0;
    v12 = *(_QWORD *)a1[1];
    v13 = (unsigned int)*a1[2] >> 3;
    v14 = *(_QWORD *)a1[3];
    while ( *(_QWORD *)(v12 + 8LL * v10) == v14 + v11 )
    {
      --v10;
      v11 += v13;
      if ( v10 == -1 )
        return 1;
    }
    return v2;
  }
  return 1;
}
