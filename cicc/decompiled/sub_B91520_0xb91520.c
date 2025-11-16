// Function: sub_B91520
// Address: 0xb91520
//
void __fastcall sub_B91520(_WORD *a1, int a2)
{
  char v2; // r12
  _WORD *v3; // rcx
  __int64 v4; // rdx
  __int64 *v5; // rbx
  __int64 v6; // r14
  __int64 *v7; // r14
  __int64 v8; // rsi

  v2 = a2;
  if ( (*(_BYTE *)a1 & 2) != 0 )
  {
    v3 = (_WORD *)*((_QWORD *)a1 - 2);
    v4 = *((unsigned int *)a1 - 2);
  }
  else
  {
    v4 = (*a1 >> 6) & 0xF;
    v3 = &a1[-4 * ((*(_BYTE *)a1 >> 2) & 0xF)];
  }
  v5 = (__int64 *)&v3[4 * v4];
  if ( a2 - (int)v4 <= 0 )
  {
    if ( a2 != (_DWORD)v4 )
    {
      v7 = &v5[-(unsigned int)(v4 - a2)];
      do
      {
        v8 = *--v5;
        if ( v8 )
          sub_B91220((__int64)v5, v8);
        *v5 = 0;
      }
      while ( v7 != v5 );
    }
  }
  else
  {
    v6 = (__int64)&v5[(unsigned int)(a2 - v4 - 1) + 1];
    do
    {
      if ( *v5 )
        sub_B91220((__int64)v5, *v5);
      *v5++ = 0;
    }
    while ( v5 != (__int64 *)v6 );
  }
  *a1 = *a1 & 0xFC3F | ((v2 & 0xF) << 6);
}
