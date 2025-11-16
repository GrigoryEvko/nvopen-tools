// Function: sub_B58EE0
// Address: 0xb58ee0
//
__int64 *__fastcall sub_B58EE0(__int64 *a1, __int64 *a2, __int64 *a3)
{
  _QWORD *v4; // rcx
  __int64 v5; // rdx
  __int64 v6; // rsi
  _QWORD *v7; // rax

  v4 = (_QWORD *)a2[1];
  v5 = *a2;
  if ( v4 != (_QWORD *)*a2 )
  {
    v6 = *a3;
    do
    {
      while ( 1 )
      {
        v7 = (_QWORD *)(v5 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (v5 & 4) != 0 )
          break;
        if ( v6 != v7[17] )
        {
          v5 = (__int64)(v7 + 18);
          if ( v4 != v7 + 18 )
            continue;
        }
        goto LABEL_8;
      }
      if ( v6 == *(_QWORD *)(*v7 + 136LL) )
        break;
      v5 = (unsigned __int64)(v7 + 1) | 4;
    }
    while ( v4 != (_QWORD *)v5 );
  }
LABEL_8:
  *a1 = v5;
  return a1;
}
