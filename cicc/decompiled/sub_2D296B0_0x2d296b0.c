// Function: sub_2D296B0
// Address: 0x2d296b0
//
void __fastcall sub_2D296B0(unsigned int *a1, __int64 a2)
{
  __int64 v2; // r13
  _QWORD *i; // rbx
  unsigned __int8 *v5; // rsi
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rsi

  v2 = *(_QWORD *)a1 + 32LL * a1[2];
  if ( *(_QWORD *)a1 != v2 )
  {
    for ( i = (_QWORD *)(*(_QWORD *)a1 + 16LL); ; i += 4 )
    {
      if ( a2 )
      {
        *(_DWORD *)a2 = *((_DWORD *)i - 4);
        *(_QWORD *)(a2 + 8) = *(i - 1);
        v5 = (unsigned __int8 *)*i;
        *(_QWORD *)(a2 + 16) = *i;
        if ( v5 )
        {
          sub_B976B0((__int64)i, v5, a2 + 16);
          *i = 0;
        }
        *(_QWORD *)(a2 + 24) = i[1];
      }
      a2 += 32;
      if ( (_QWORD *)v2 == i + 2 )
        break;
    }
    v6 = *(_QWORD *)a1;
    v7 = *(_QWORD *)a1 + 32LL * a1[2];
    if ( *(_QWORD *)a1 != v7 )
    {
      do
      {
        v8 = *(_QWORD *)(v7 - 16);
        v7 -= 32;
        if ( v8 )
          sub_B91220(v7 + 16, v8);
      }
      while ( v7 != v6 );
    }
  }
}
