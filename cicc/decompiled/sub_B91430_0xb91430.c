// Function: sub_B91430
// Address: 0xb91430
//
void __fastcall sub_B91430(_BYTE *a1, __int64 a2)
{
  __int64 v2; // rbx
  _BYTE *v3; // r12
  __int64 v4; // rsi
  __int64 v5; // r13
  __int64 v6; // r12

  v2 = (__int64)a1;
  if ( (*a1 & 2) != 0 )
  {
    v5 = *((_QWORD *)a1 - 2);
    v6 = v5 + 8LL * *((unsigned int *)a1 - 2);
    if ( v5 != v6 )
    {
      do
      {
        a2 = *(_QWORD *)(v6 - 8);
        v6 -= 8;
        if ( a2 )
          sub_B91220(v6, a2);
      }
      while ( v5 != v6 );
      v6 = *((_QWORD *)a1 - 2);
    }
    if ( (_BYTE *)v6 != a1 )
      _libc_free(v6, a2);
  }
  else
  {
    v3 = &a1[-8 * ((*a1 >> 2) & 0xF)];
    while ( (_BYTE *)v2 != v3 )
    {
      while ( 1 )
      {
        v4 = *(_QWORD *)(v2 - 8);
        v2 -= 8;
        if ( !v4 )
          break;
        sub_B91220(v2, v4);
        if ( (_BYTE *)v2 == v3 )
          return;
      }
    }
  }
}
