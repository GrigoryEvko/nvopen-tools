// Function: sub_7D4B20
// Address: 0x7d4b20
//
void __fastcall sub_7D4B20(__int64 a1, _QWORD *a2, _QWORD *a3, __int64 a4, __int64 *a5)
{
  _QWORD *v5; // r15
  _QWORD *v7; // r13
  __int64 j; // rax
  _QWORD *v10; // rbx
  __int64 *v11; // rax
  __int64 v12; // rbx
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // r13
  unsigned __int64 v16; // rax
  __int64 *v17; // rax
  __int64 v18; // rsi
  __int64 i; // [rsp+8h] [rbp-38h]

  v5 = a2;
  v7 = a3;
  for ( i = a4; v7; v7 = (_QWORD *)*v7 )
  {
    for ( j = v7[1]; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    v10 = *(_QWORD **)(*(_QWORD *)(*(_QWORD *)j + 96LL) + 144LL);
    if ( v10 )
    {
      while ( *v10 != *(_QWORD *)a1 )
      {
        v10 = (_QWORD *)v10[1];
        if ( !v10 )
          goto LABEL_9;
      }
      v11 = (__int64 *)((__int64 (*)(void))sub_878440)();
      v11[1] = (__int64)v10;
      a4 = *a5;
      *v11 = *a5;
      *a5 = (__int64)v11;
    }
LABEL_9:
    ;
  }
  v12 = 1182720;
  if ( a2 )
  {
    do
    {
      v18 = v5[1];
      if ( (*(_BYTE *)(a1 + 17) & 0x40) == 0 )
      {
        *(_BYTE *)(a1 + 16) &= ~0x80u;
        *(_QWORD *)(a1 + 24) = 0;
      }
      if ( v18 )
      {
        v13 = a1;
        v15 = sub_7D4A40((__int64 *)a1, v18, 0x280000u, a4, (__int64)a5);
      }
      else
      {
        v18 = a1;
        v13 = *(_QWORD *)(i + 8);
        v15 = sub_7D4600(v13, (__int64 *)a1, 0x280000u, a4, (__int64)a5);
      }
      if ( v15 )
      {
        v16 = *(unsigned __int8 *)(v15 + 80);
        a4 = v15;
        if ( (_BYTE)v16 == 16 )
        {
          a4 = **(_QWORD **)(v15 + 88);
          v16 = *(unsigned __int8 *)(a4 + 80);
        }
        if ( (_BYTE)v16 == 24 )
          v16 = *(unsigned __int8 *)(*(_QWORD *)(a4 + 88) + 80LL);
        if ( (unsigned __int8)v16 <= 0x14u )
        {
          if ( _bittest64(&v12, v16) )
          {
            v17 = (__int64 *)sub_878440(v13, v18, v14, a4);
            v17[1] = v15;
            *v17 = *a5;
            *a5 = (__int64)v17;
          }
        }
      }
      v5 = (_QWORD *)*v5;
    }
    while ( v5 );
  }
}
