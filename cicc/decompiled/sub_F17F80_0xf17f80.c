// Function: sub_F17F80
// Address: 0xf17f80
//
__int64 __fastcall sub_F17F80(__int64 a1, unsigned __int64 *a2)
{
  _QWORD *v2; // r12
  __int64 result; // rax
  __int64 v4; // r13
  unsigned __int64 v6; // rax
  _QWORD *v7; // r12
  _QWORD *v8; // rbx

  v2 = *(_QWORD **)a1;
  result = 3LL * *(unsigned int *)(a1 + 8);
  v4 = *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v4 )
  {
    do
    {
      if ( a2 )
      {
        *a2 = 6;
        a2[1] = 0;
        v6 = v2[2];
        a2[2] = v6;
        if ( v6 != 0 && v6 != -4096 && v6 != -8192 )
          sub_BD6050(a2, *v2 & 0xFFFFFFFFFFFFFFF8LL);
      }
      v2 += 3;
      a2 += 3;
    }
    while ( (_QWORD *)v4 != v2 );
    v7 = *(_QWORD **)a1;
    result = 3LL * *(unsigned int *)(a1 + 8);
    v8 = (_QWORD *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8));
    if ( *(_QWORD **)a1 != v8 )
    {
      do
      {
        result = *(v8 - 1);
        v8 -= 3;
        if ( result != 0 && result != -4096 && result != -8192 )
          result = sub_BD60C0(v8);
      }
      while ( v8 != v7 );
    }
  }
  return result;
}
