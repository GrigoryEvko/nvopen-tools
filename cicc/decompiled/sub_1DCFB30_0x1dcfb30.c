// Function: sub_1DCFB30
// Address: 0x1dcfb30
//
__int64 __fastcall sub_1DCFB30(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax
  int v3; // r13d
  unsigned int i; // ebx
  __int64 v6; // rdi
  __int64 v7; // r9
  __int64 v8; // rcx
  unsigned int v9; // r8d
  _WORD *v10; // rdx
  unsigned __int16 v11; // cx
  __int16 *v12; // rdx
  __int16 v13; // ax

  result = a1[45];
  v3 = *(_DWORD *)(result + 16);
  if ( v3 != 1 )
  {
    for ( i = 1; i != v3; ++i )
    {
      while ( 1 )
      {
        v6 = a1[46];
        if ( *(_QWORD *)(v6 + 8LL * i) || (result = a1[49], *(_QWORD *)(result + 8LL * i)) )
        {
          v7 = *(_QWORD *)(a2 + 24);
          result = *(unsigned int *)(v7 + 4LL * (i >> 5));
          if ( !_bittest((const int *)&result, i) )
            break;
        }
        if ( v3 == ++i )
          return result;
      }
      v8 = a1[45];
      if ( !v8 )
        BUG();
      v9 = i;
      v10 = (_WORD *)(*(_QWORD *)(v8 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v8 + 8) + 24LL * i + 8));
      if ( *v10 )
      {
        v11 = i + *v10;
        v12 = v10 + 1;
        do
        {
          if ( (*(_QWORD *)(v6 + 8LL * v11) || *(_QWORD *)(a1[49] + 8LL * v11))
            && ((*(_DWORD *)(v7 + 4LL * (v11 >> 5)) >> v11) & 1) == 0 )
          {
            v9 = v11;
          }
          v13 = *v12++;
          v11 += v13;
        }
        while ( v13 );
      }
      result = sub_1DCE640((__int64)a1, v9, 0);
    }
  }
  return result;
}
