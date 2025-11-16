// Function: sub_876830
// Address: 0x876830
//
void __fastcall sub_876830(__int64 a1)
{
  _QWORD *v1; // rbx
  _QWORD *i; // r12
  __int64 v3; // rsi
  _QWORD *v4; // rax
  _QWORD *v5; // rdi
  _QWORD *v6; // rax
  _QWORD *v7; // rax
  _QWORD *v8; // rbx
  __int64 v9; // rdi

  v1 = *(_QWORD **)(a1 + 8);
  for ( i = *(_QWORD **)(a1 + 88); v1; v1 = (_QWORD *)*v1 )
  {
    v3 = v1[1];
    if ( v3 )
      sub_8756F0(1, v3, (_QWORD *)(v3 + 48), 0);
    v4 = (_QWORD *)v1[6];
    if ( v4 )
    {
      if ( i )
      {
        if ( v4 == i )
        {
          v6 = (_QWORD *)*i;
          v5 = i;
          i = (_QWORD *)*i;
        }
        else
        {
          v5 = i;
          do
            v5 = (_QWORD *)*v5;
          while ( v4 != v5 );
          *(_QWORD *)v5[1] = *v5;
          v6 = (_QWORD *)*v5;
        }
        if ( v6 )
          v6[1] = v5[1];
        sub_869F90(v5);
      }
      v1[6] = 0;
    }
  }
  if ( i )
  {
    v7 = (_QWORD *)i[1];
    v8 = (_QWORD *)*i;
    if ( !v7 )
      goto LABEL_19;
    while ( 1 )
    {
      *v7 = v8;
LABEL_19:
      if ( !v8 )
        break;
      while ( 1 )
      {
        v9 = (__int64)i;
        v8[1] = i[1];
        i[1] = 0;
        *i = 0;
        i = v8;
        sub_869970(v9);
        v7 = (_QWORD *)v8[1];
        v8 = (_QWORD *)*v8;
        if ( v7 )
          break;
        if ( !v8 )
          goto LABEL_22;
      }
    }
LABEL_22:
    i[1] = 0;
    *i = 0;
    sub_869970((__int64)i);
  }
}
