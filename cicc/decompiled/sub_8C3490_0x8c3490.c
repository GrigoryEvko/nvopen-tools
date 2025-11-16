// Function: sub_8C3490
// Address: 0x8c3490
//
void __fastcall sub_8C3490(__int64 a1)
{
  __int64 v1; // r13
  _QWORD *v2; // rbx
  _QWORD *i; // r12
  __int64 v4; // r14
  _QWORD *v5; // rax
  _QWORD *j; // r12
  __int64 v7; // r14
  _QWORD *v8; // rax
  __int64 v9; // rdi

  if ( a1 )
  {
    v1 = a1;
    do
    {
      while ( 1 )
      {
        if ( (unsigned __int8)(*(_BYTE *)(v1 + 140) - 9) <= 2u )
        {
          v2 = *(_QWORD **)(v1 + 168);
          for ( i = (_QWORD *)v2[18]; i; i = (_QWORD *)*i )
          {
            v4 = *(_QWORD *)(i[1] + 168LL);
            v5 = sub_727DC0();
            v5[1] = v1;
            *v5 = *(_QWORD *)(v4 + 128);
            *(_QWORD *)(v4 + 128) = v5;
          }
          for ( j = (_QWORD *)v2[17]; j; j = (_QWORD *)*j )
          {
            v7 = j[1];
            v8 = sub_727DC0();
            v8[1] = v1;
            *v8 = *(_QWORD *)(v7 + 232);
            *(_QWORD *)(v7 + 232) = v8;
          }
          v9 = v2[19];
          if ( v9 )
            break;
        }
        v1 = *(_QWORD *)(v1 + 112);
        if ( !v1 )
          return;
      }
      sub_8C3410(v9);
      v1 = *(_QWORD *)(v1 + 112);
    }
    while ( v1 );
  }
}
